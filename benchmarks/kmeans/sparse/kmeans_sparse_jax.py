import gzip
import json
from functools import partial
from pathlib import Path
from time import time_ns

import futhark_data
import jax
import numpy as np
import torch
from benchmark import Benchmark
from jax import grad
from jax import numpy as jnp
from jax.experimental import sparse
from jax.experimental.sparse import sparsify
from jax.lax import while_loop
from scipy.sparse import csr_matrix

data_dir = Path(__file__).parent / "data"


class KMeansSparse(Benchmark):
    def __init__(self, name, runs):
        self.runs = runs
        self.name = name
        self.max_iter = 10
        self.k = 10
        self.kind = "jax"

    def prepare(self):
        sp_data = data_gen(self.name)

        coo = csr_matrix(sp_data).tocoo()
        values = coo.data
        indices = np.transpose(np.vstack((coo.row, coo.col)))
        shape = coo.shape

        self.features = sparse.BCOO((values, indices), shape=shape)
        self.clusters = jnp.array(get_clusters(self.k, *sp_data, shape[1]).detach())

    def calculate_objective(self, runs):
        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            self.objective = jax.block_until_ready(
                kmeans(self.max_iter, self.clusters, self.features)
            )
            timings[i] = (time_ns() - start) / 1000
        return timings

    def calculate_jacobian(self, runs):
        return np.zeros(runs + 1)

    def validate(self):
        return
        #data_file = data_dir / f"{self.name}.out"
        #if data_file.exists():
        #    out = tuple(futhark_data.load(open(data_file)))[0]
        #    assert np.allclose(
        #        out, self.objective, rtol=1e-02, atol=1e-02
        #    )


def get_clusters(k, values, indices, pointers, num_col):
    end = pointers[k]
    sp_clusters = torch.sparse_csr_tensor(
        pointers[: (k + 1)],
        indices[:end],
        values[:end],
        requires_grad=True,
        dtype=torch.float32,
        size=(k, num_col),
    ).to_dense()
    return sp_clusters


def bench_all(runs, output, datasets=["movielens", "nytimes", "scrna"], prec="f32"):
    times = {}
    for data in datasets:
        kmeans = KMeansSparse(data, runs)
        kmeans.benchmark()
        times["data/" + data] = {
            kmeans.kind: {
                "objective": kmeans.objective_time,
                "objective_std": kmeans.objective_std,
            }
        }
    with open(output, "w") as f:
        json.dump(times, f, indent=2)
    return


def cost(points, centers):
    a_sqr = jnp.sum(points * points, 1)[None, :]
    b_sqr = jnp.sum(centers * centers, 1)[:, None]
    diff = jnp.matmul(points, centers.T).T
    dists = sparse.todense(a_sqr) + sparse.todense(b_sqr) - 2 * sparse.todense(diff)
    min_dist = jnp.min(dists, axis=0)
    return min_dist.sum()


@jax.jit
def kmeans(max_iter, clusters, features):
    cost_sp = sparsify(cost)

    def cond(v):
        t, _rmse, _ = v
        return t < max_iter

    def body(v):
        t, rmse, clusters = v
        f_vjp = grad(partial(cost_sp, features))
        d, hes = jvp(f_vjp, [clusters], [jnp.ones(shape=clusters.shape)])
        new_cluster = clusters - d / hes
        rmse = ((new_cluster - clusters) ** 2).sum()
        return t + 1, rmse, new_cluster

    _t, _rmse, clusters = while_loop(cond, sparsify(body), (0, float("inf"), clusters))
    return clusters


def data_gen(name):
    """Dataformat CSR  (https://en.wikipedia.org/wiki/Sparse_matrix)"""
    data_file = data_dir / f"{name}.in.gz"
    assert data_file.exists()

    with gzip.open(data_file.resolve(), ("rb")) as f:
        values, indices, pointers = tuple(futhark_data.load(f))

    return (values, indices, pointers)
