import gzip
import json
from pathlib import Path
from time import time_ns

import futhark_data
import jax
import jax.numpy as jnp
import numpy as np
from benchmark import Benchmark
from jax import grad, jvp, jit, vmap

data_dir = Path(__file__).parent / "data"

print(f"Running on: {jax.default_backend()}")


class KMeans(Benchmark):
    def __init__(self, name, runs, kind, kmeans_fn):
        self.runs = runs
        self.name = name
        self.kind = kind
        self.kmeans_fn = kmeans_fn

    def prepare(self):
        _, k, max_iter, features = data_gen(self.name)
        self.max_iter = max_iter
        self.features = features
        self.clusters = jnp.flip(features[-int(k) :], (0,))

    def calculate_objective(self, runs):
        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            _t, _rmse, self.objective = jax.block_until_ready(
                self.kmeans_fn(self.max_iter, self.clusters, self.features))
            timings[i] = (time_ns() - start) / 1000
        return timings

    def calculate_jacobian(self, runs):
        return np.zeros(runs + 1)

    def validate(self):
        data_file = data_dir / f"{self.name}.out"
        if data_file.exists():
            out = tuple(futhark_data.load(open(data_file, "rb")))
            assert np.allclose(out, self.objective, rtol=1e-02, atol=1e-05)


def bench_all(runs, output, datasets=["kdd_cup", "random"], prec="f32"):
    times = {}
    for data in datasets:
        _kmeans = KMeans(data, runs, "jax", kmeans)
        _kmeans_vmap = KMeans(data, runs, "jax-vmap", kmeans_vmap)
        _kmeans.benchmark()
        _kmeans_vmap.benchmark()
        times["data/" + data] = {
            _kmeans.kind: {
                "objective": _kmeans.objective_time,
                "objective_std": _kmeans.objective_std,
            },
            _kmeans_vmap.kind: {
                "objective": _kmeans_vmap.objective_time,
                "objective_std": _kmeans_vmap.objective_std,
            },
        }
    with open(output, "w") as f:
        json.dump(times, f, indent=2)
    return


def cost(points, centers):
    def all_pairs_norm(a, b):
        a_sqr = jnp.sum(a**2, 1)[None, :]
        b_sqr = jnp.sum(b**2, 1)[:, None]
        diff = jnp.matmul(a, b.T).T
        return a_sqr + b_sqr - 2 * diff

    dists = all_pairs_norm(points, centers)
    min_dist = jnp.min(dists, axis=0)
    return min_dist.sum()


def euclid_dist(xs,ys):
  return (vmap(lambda x, y: (x - y) * (x - y))(xs, ys)).sum()

def cost_vmap(features, clusters):
    dists = vmap(
        lambda feature: vmap(
            lambda cluster: euclid_dist(feature, cluster)
        )(clusters)
    )(features)
    min_dist = jnp.min(dists, axis=1)
    return min_dist.sum()


@jit
def kmeans(max_iter, clusters, features, _tolerance=1):
    tolerance = 1.0

    def cond(v):
        t, rmse, _ = v
        return jnp.logical_and(t < max_iter, rmse > tolerance)

    def body(v):
        t, rmse, clusters = v
        f_diff = grad(lambda cs: cost(features,cs))
        d, hes = jvp(f_diff, [clusters], [jnp.ones(shape=clusters.shape)])
        new_cluster = clusters - d / hes
        rmse = ((new_cluster - clusters) ** 2).sum()
        return t + 1, rmse, new_cluster

    return jax.lax.while_loop(cond, body, (0, float("inf"), clusters))

@jit
def kmeans_vmap(max_iter, clusters, features, _tolerance=1):
    tolerance = 1.0

    def cond(v):
        t, rmse, _ = v
        return jnp.logical_and(t < max_iter, rmse > tolerance)

    def body(v):
        t, rmse, clusters = v
        f_diff = grad(lambda cs: cost_vmap(features,cs))
        d, hes = jvp(f_diff, [clusters], [jnp.ones(shape=clusters.shape)])
        x = vmap(lambda ds, hs: vmap(lambda _d, _h: _d/_h)(ds, hs))(d, hes)
        new_cluster = vmap(lambda cs, xs: vmap(lambda c, _x: c - _x)(cs, xs))(clusters, x)
        rmse = (vmap(lambda new, old: euclid_dist(new, old))(new_cluster, clusters)).sum()
        return t + 1, rmse, new_cluster

    return jax.lax.while_loop(cond, body, (0, float("inf"), clusters))


def data_gen(name):
    data_file = data_dir / f"{name}.in.gz"
    assert data_file.exists()
    kmeans_args = tuple(futhark_data.load(gzip.open(data_file)))
    return tuple(map(jnp.array, kmeans_args))
