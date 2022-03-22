import argparse
from functools import partial
from pathlib import Path
from time import time_ns

import futhark_data
import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jvp, vjp, grad

from benchmark import (Benchmark, set_precision)
import json
import gzip

data_dir = Path(__file__).parent / 'data'


class KMeans(Benchmark):
    def __init__(self, name, runs):
        self.runs = runs
        self.name = name
        self.kind = "jax"

    def prepare(self):
        _, k, max_iter, features = data_gen(self.name)
        self.max_iter = max_iter
        self.features = features
        self.clusters = jnp.flip(features[-int(k):], (0,))

    def calculate_objective(self, runs):
      timings = np.zeros(runs + 1)
      for i in range(runs + 1):
          start = time_ns()
          #self.objective = kmeans(self.max_iter, self.clusters, self.features).block_until_ready()
          t, rmse, self.objective = jax.block_until_ready(kmeans(self.max_iter, self.clusters, self.features))
          timings[i] = (time_ns() - start)/1000
          print(self.name)
          print(f't: {t}, rmse: {rmse}, max_iter: {self.max_iter}')
      return timings

    def calculate_jacobian(self, runs):
        return np.zeros(runs + 1)

    def validate(self):
        data_file = data_dir / f'{self.name}.out'
        if data_file.exists():
          out = tuple(futhark_data.load(open(data_file)))[0]
          assert(np.allclose(out, self.objective, rtol=1e-02, atol=1e-05))

def bench(kmeans_args, times=10):
    timings = np.zeros((times,))
    _, k, max_iter, features = kmeans_args
    clusters = jnp.flip(features[-int(k):], (0,))
    for i in range(times):
        start = time_ns()
        kmeans(max_iter, clusters, features).block_until_ready()
        timings[i] += time_ns() - start

    return float(timings[1:].mean()), float(timings[1:].std())

def benchmarks(datasets = ["kdd_cup", "random"], runs=10, output="kmeans_jax.json"):
  times = {}
  for data in datasets:
    kmeans = KMeans(data, runs)
    kmeans.benchmark()
    kmeans.validate()
    times['data/' + data] = { 'jax' : 
                           { 'objective': kmeans.objective_time,
                             'objective_std': kmeans.objective_std
                           }
                  }
  with open(output,'w') as f:
    json.dump(times, f, indent=2)
  print("Benchmarks output to: " + output)
  return

def cost(points, centers):
    def all_pairs_norm(a, b):
        a_sqr = jnp.sum(a ** 2, 1)[None, :]
        b_sqr = jnp.sum(b ** 2, 1)[:, None]
        diff = jnp.matmul(a, b.T).T
        return a_sqr + b_sqr - 2 * diff

    dists = all_pairs_norm(points, centers)
    min_dist = jnp.min(dists, axis=0)
    return min_dist.sum()


@jax.jit
def kmeans(max_iter, clusters, features, _tolerance=1):
    tolerance=1
    def cond(v):
        t, rmse, _ = v
        return jnp.logical_and(t < max_iter, rmse > tolerance)

    def body(v):
        t, rmse, clusters = v
        f_vjp = grad(partial(cost, features))
        hes = grad(lambda x: jnp.vdot(f_vjp(x), jnp.ones(shape=clusters.shape)))(clusters)
        new_cluster = clusters - f_vjp(clusters) / hes
        rmse = ((new_cluster - clusters) ** 2).sum()
        return t + 1, rmse, new_cluster

    t, rmse, clusters = jax.lax.while_loop(cond, body, (0, float("inf"), clusters))

    return t, rmse, clusters

def data_gen(name):
    data_file = data_dir / f'{name}.in.gz'
    assert data_file.exists()
    kmeans_args = tuple(futhark_data.load(gzip.open(data_file)))
    return tuple(map(jnp.array, kmeans_args))

def bench(kmeans_args, times=10):
    timings = np.zeros((times,))
    _, k, max_iter, features = kmeans_args
    clusters = jnp.flip(features[-int(k):], (0,))
    for i in range(times):
        start = time_ns()
        kmeans(max_iter, clusters, features).block_until_ready()
        timings[i] += time_ns() - start

    return float(timings[1:].mean()), float(timings[1:].std())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Newtonian KMeans"
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--datasets", nargs="+", default=["kdd_cup"])  # , "random"])

    args = parser.parse_args()

    for name in args.datasets:
        print(name, '±'.join(map(str, bench(data_gen(name)))), 'micro seconds')
