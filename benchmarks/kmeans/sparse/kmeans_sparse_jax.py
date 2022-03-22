from functools import partial
from pathlib import Path
import gzip
import futhark_data
from scipy.sparse import csr_matrix
import torch

import jax
from jax import numpy as jnp, jacrev, jacfwd
from jax.experimental import sparse
from jax.experimental.sparse import sparsify
from jax.lax import while_loop
from jax.random import PRNGKey, split, normal, bernoulli
from benchmark import (Benchmark, set_precision)
import numpy as np
from time import time_ns

data_dir = Path(__file__).parent / "data"

class KMeansSparse(Benchmark):
    def __init__(self, name, runs, max_iter = 10, threshold=5e-3, k=10):
        self.runs = runs
        self.name = name
        self.max_iter = max_iter
        self.threshold = threshold
        self.k = k
        self.kind = "jax"

    def prepare(self):
      sp_data = data_gen(self.name)

      coo = csr_matrix(sp_data).tocoo()
      values = coo.data
      indices = np.transpose(np.vstack((coo.row, coo.col)))
      shape = coo.shape

      end = sp_data[2][self.k]
      cluster_shape = (self.k, shape[1])
      self.features = sparse.BCOO((values, indices), shape=shape)
      self.clusters = jnp.array(get_clusters(self.k, *sp_data, shape[1]).detach())

    def calculate_objective(self, runs):
      timings = np.zeros(runs + 1)
      for i in range(runs + 1):
          start = time_ns()
          self.objective = (kmeans(self.max_iter, self.clusters, self.features)).block_until_ready()
          timings[i] = (time_ns() - start)/1000
      return timings

    def calculate_jacobian(self, runs):
        return np.zeros(runs + 1)

    def validate(self):
        data_file = data_dir / f'{self.name}.out'
        if data_file.exists():
          out = tuple(futhark_data.load(open(data_file)))[0]
          assert(np.allclose(out, self.objective.cpu().detach().numpy(), rtol=1e-02, atol=1e-05))

def get_clusters(k, values, indices, pointers, num_col):
    end = pointers[k]
    sp_clusters = (
        torch.sparse_csr_tensor(
            pointers[: (k + 1)],
            indices[:end],
            values[:end],
            requires_grad=True,
            dtype=torch.float32,
            size=(k, num_col),
        ).to_dense()
    )
    return sp_clusters

def benchmarks(datasets = ['movielens', 'nytimes', 'scrna'], runs=10, output="kmeans_sparse_pytorch.json"):
  times = {}
  for data in datasets:
    kmeans = KMeansSparse(data, runs)
    kmeans.benchmark()
    times['data/' + data] = { kmeans.kind : 
                            { 'objective': kmeans.objective_time,
                              'objective_std': kmeans.objective_std,
                             }
                  }
  with open(output,'w') as f:
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
def kmeans(max_iter, clusters, features, tolerance=1):
    cost_sp = sparsify(cost)

    def cond(v):
        t, rmse, _ = v
        return jnp.logical_and(t < max_iter, rmse > tolerance)

    def body(v):
        t, rmse, clusters = v
        jac_fn = jacrev(partial(cost_sp, features), allow_int=True)
        hes_fn = jacfwd(jac_fn)

        new_cluster = clusters - jac_fn(clusters) / hes_fn(clusters).sum((0, 1))
        rmse = ((new_cluster - clusters) ** 2).sum()
        return t + 1, rmse, new_cluster

    t, rmse, clusters = while_loop(cond, sparsify(body), (0, float("inf"), clusters))
    return clusters

def data_gen(name):
    """Dataformat CSR  (https://en.wikipedia.org/wiki/Sparse_matrix)"""
    data_file = data_dir / f"{name}.in.gz"
    assert data_file.exists()

    with gzip.open(data_file.resolve(), ("rb")) as f:
        values, indices, pointers = tuple(futhark_data.load(f))

    return (values, indices, pointers)

if __name__ == '__main__':
    data_key, sparse_key = split(PRNGKey(8))
    num_datapoints = 100
    num_features = 7
    sparsity = .5
    k = 5
    max_iter = 10
    features = normal(data_key, (num_datapoints, num_features))
    features = features * bernoulli(sparse_key, sparsity, (num_datapoints, num_features))
    clusters = features[:k]
    features = sparse.BCOO.fromdense(features)
    new_cluster = kmeans(max_iter, clusters, features)
    print(new_cluster)
