import gzip
import json
from functools import partial
from pathlib import Path
from time import time_ns

import futhark_data
import jax
import jax.numpy as jnp
import numpy as np
from benchmark import Benchmark
from jax import grad

data_dir = Path(__file__).parent / "data"


class KMeans(Benchmark):
    def __init__(self, name, runs):
        self.runs = runs
        self.name = name
        self.kind = "jax"

    def prepare(self):
        _, k, max_iter, features = data_gen(self.name)
        self.max_iter = max_iter
        self.features = features
        self.clusters = jnp.flip(features[-int(k) :], (0,))

    def calculate_objective(self, runs):
        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            kmeans_fn = jax.jit(partial(kmeans, cost))
            _t, _rmse, self.objective = jax.block_until_ready(
                kmeans_fn(self.max_iter, self.clusters, self.features)
            )
            timings[i] = (time_ns() - start) / 1000
        return timings

    def calculate_jacobian(self, runs):
        return np.zeros(runs + 1)

    def validate(self):
        data_file = data_dir / f"{self.name}.out"
        if data_file.exists():
            out = tuple(futhark_data.load(open(data_file, "rb")))
            assert np.allclose(
                out, self.objective, rtol=1e-02, atol=1e-05
            )


class KMeansVMap(Benchmark):
    def __init__(self, name, runs):
        self.runs = runs
        self.name = name
        self.kind = "jax-vmap"

    def prepare(self):
        _, k, max_iter, features = data_gen(self.name)
        self.max_iter = max_iter
        self.features = features
        self.clusters = jnp.flip(features[-int(k) :], (0,))

    def calculate_objective(self, runs):
        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            kmeans_fn = jax.jit(partial(kmeans, cost_vmap))
            _t, _rmse, self.objective = jax.block_until_ready(
                kmeans_fn(self.max_iter, self.clusters, self.features)
            )
            timings[i] = (time_ns() - start) / 1000
        return timings

    def calculate_jacobian(self, runs):
        return np.zeros(runs + 1)

    def validate(self):
        data_file = data_dir / f"{self.name}.out"
        if data_file.exists():
            out = tuple(futhark_data.load(open(data_file, "rb")))
            assert np.allclose(
                out, self.objective, rtol=1e-02, atol=1e-05
            )


def bench_all(runs, output, datasets=["kdd_cup", "random"], prec="f32"):
    times = {}
    for data in datasets:
        kmeans = KMeans(data, runs)
        kmeans_vmap = KMeansVMap(data, runs)
        kmeans.benchmark()
        kmeans_vmap.benchmark()
        times["data/" + data] = {
            kmeans.kind: {
                "objective": kmeans.objective_time,
                "objective_std": kmeans.objective_std,
            },
            kmeans_vmap.kind: {
                "objective": kmeans_vmap.objective_time,
                "objective_std": kmeans_vmap.objective_std,
            }
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

def cost_vmap(features, clusters):
    dists = jax.vmap(
        lambda feature: jax.vmap(
            lambda cluster: jnp.dot((feature - cluster), (feature - cluster))
        )(clusters)
    )(features)
    min_dist = jnp.min(dists, axis=1)
    return min_dist.sum()


def kmeans(cost_fn, max_iter, clusters, features, _tolerance=1):
    tolerance = 1.0

    def cond(v):
        t, rmse, _ = v
        return jnp.logical_and(t < max_iter, rmse > tolerance)

    def body(v):
        t, rmse, clusters = v
        f_vjp = grad(partial(cost_fn, features))
        hes = grad(lambda x: jnp.vdot(f_vjp(x), jnp.ones(shape=clusters.shape)))(
            clusters
        )
        new_cluster = clusters - f_vjp(clusters) / hes
        rmse = ((new_cluster - clusters) ** 2).sum()
        return t + 1, rmse, new_cluster

    return jax.lax.while_loop(cond, body, (0, float("inf"), clusters))


def data_gen(name):
    data_file = data_dir / f"{name}.in.gz"
    assert data_file.exists()
    kmeans_args = tuple(futhark_data.load(gzip.open(data_file)))
    return tuple(map(jnp.array, kmeans_args))
