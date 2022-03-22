import gzip
from functools import partial
from pathlib import Path
import sys

import futhark_data
import torch
from torch.autograd.functional import vhp
from torch.autograd.functional import vjp
from scipy.sparse import csr_matrix
import numpy as np

from benchmark import (Benchmark, set_precision)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

data_dir = Path(__file__).parent / "data"

VERBOSE = False

class KMeansSparse(Benchmark):
    def __init__(self, name, runs, max_iter = 10, threshold=5e-3, k=10):
        self.runs = runs
        self.name = name
        self.max_iter = max_iter
        self.threshold = threshold
        self.k = k
        self.kind = "pytorch"

    def prepare(self):
      sp_data = data_gen(self.name)

      coo = csr_matrix(sp_data).tocoo()
      values = coo.data
      indices = np.vstack((coo.row, coo.col))

      i = torch.LongTensor(indices)
      v = torch.FloatTensor(values)
      shape = coo.shape

      self.sp_features = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)
      self.sp_clusters = get_clusters(self.k, *sp_data, self.sp_features.size()[1]).to(device)
      print(self.sp_clusters.detach())
      sys.exit(1)


    def calculate_objective(self):
       self.objective = kmeans(self.max_iter, self.sp_clusters, self.sp_features, self.threshold)

    def calculate_jacobian(self):
        return

    def validate(self):
        data_file = data_dir / f'{self.name}.out'
        if data_file.exists():
          out = tuple(futhark_data.load(open(data_file)))[0]
          assert(np.allclose(out, self.objective.cpu().detach().numpy(), rtol=1e-02, atol=1e-05))

def benchmarks(datasets = ['movielens', 'nytimes', 'scrna'], runs=10, output="kmeans_sparse_pytorch.json"):
  set_precision("f32")
  times = {}
  for data in datasets:
    kmeans = KMeansSparse(data, runs)
    kmeans.benchmark()
    times['data/' + data] = { 'pytorch' : 
                            { 'objective': kmeans.objective_time,
                              'objective_std': kmeans.objective_std,
                             }
                  }
  with open(output,'w') as f:
    json.dump(times, f, indent=2)
  return

def all_pairs_norm(a, b):
    a_sqr = torch.sparse.sum(a ** 2, 1).to_dense()[None, :]
    b_sqr = torch.sum(b ** 2, 1)[:, None]
    diff = torch.sparse.mm(a, b.T).T
    if VERBOSE:
        print('entries', diff.numel(), 'zeros', diff.numel() - torch.count_nonzero(diff))
    return a_sqr + b_sqr - 2 * diff


def cost(points, centers):
    dists = all_pairs_norm(points, centers)
    (min_dist, _) = torch.min(dists, dim=0)
    return min_dist.sum()


def kmeans(max_iter, clusters, features, tolerance=1):
    t = 0
    converged = False
    hes_v = torch.ones_like(clusters)
    while t < max_iter and not converged:
        _, jac = vjp(partial(cost, features), clusters, v=torch.tensor(1.0))
        _, hes = vhp(partial(cost, features), clusters, v=hes_v)

        new_cluster = clusters - jac / hes
        converged = ((new_cluster - clusters) ** 2).sum() < tolerance
        clusters = new_cluster
        t += 1
    return clusters


def data_gen(name):
    """Dataformat CSR  (https://en.wikipedia.org/wiki/Sparse_matrix)"""
    data_file = data_dir / f"{name}.in.gz"
    assert data_file.exists()

    with gzip.open(data_file.resolve(), ("rb")) as f:
        values, indices, pointers = tuple(futhark_data.load(f))

    return (values, indices, pointers)

def bench_gpu(dataset, k=10, max_iter=10, times=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = torch.zeros((times,), dtype=float)
    sp_data = data_gen(dataset)

    coo = csr_matrix(sp_data).tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    sp_features = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    sp_clusters = get_clusters(k, *sp_data, sp_features.size()[1])
    for i in range(times):
        torch.cuda.synchronize()
        start.record()
        kmeans(max_iter, sp_clusters, sp_features, threshold)
        torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        timings[i] = start.elapsed_time(end) * 1000  # micro seconds

    return float(timings[1:].mean()), float(timings[1:].std())


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


if __name__ == "__main__":
    k = 10
    max_iter = 10
    threshold = 5e-3

    for dataset in ['movielens', 'nytimes', 'scrna']:
        print(dataset, f'{bench_gpu(dataset, max_iter=max_iter)} micro seconds')
