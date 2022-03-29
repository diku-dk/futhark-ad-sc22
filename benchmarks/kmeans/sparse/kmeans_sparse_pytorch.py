import gzip
import json
from functools import partial
from pathlib import Path

import futhark_data
import numpy as np
import torch
from benchmark import Benchmark, set_precision
from scipy.sparse import csr_matrix
from torch.autograd.functional import vhp, vjp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_dir = Path(__file__).parent / "data"


class KMeansSparse(Benchmark):
    def __init__(self, name, runs):
        self.runs = runs
        self.name = name
        self.max_iter = 10
        self.k = 10
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
        self.sp_clusters = get_clusters(
            self.k, *sp_data, self.sp_features.size()[1]
        ).to(device)

    def calculate_objective(self):
        self.objective = kmeans(
            self.max_iter,
            self.sp_clusters,
            self.sp_features,
        )

    def calculate_jacobian(self):
        return

    def validate(self):
        data_file = data_dir / f"{self.name}.out"
        if data_file.exists():
            out = tuple(futhark_data.load(open(data_file, "rb")))[0]
            assert np.allclose(
                out, self.objective.cpu().detach().numpy(), rtol=1e-02, atol=1e-02
            )
        print(f"{self.kind}: validates on {self.name}")


def bench_all(runs, output, datasets=["movielens", "nytimes", "scrna"], prec="f32"):
    set_precision("f32")
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


def all_pairs_norm(a, b):
    a_sqr = torch.sparse.sum(a**2, 1).to_dense()[None, :]
    b_sqr = torch.sum(b**2, 1)[:, None]
    diff = torch.sparse.mm(a, b.T).T
    return a_sqr + b_sqr - 2 * diff


def cost(points, centers):
    dists = all_pairs_norm(points, centers)
    (min_dist, _) = torch.min(dists, dim=0)
    return min_dist.sum()


def kmeans(max_iter, clusters, features):
    t = 0
    hes_v = torch.ones_like(clusters)
    while t < max_iter:
        _, jac = vjp(partial(cost, features), clusters, v=torch.tensor(1.0))
        _, hes = vhp(partial(cost, features), clusters, v=hes_v)

        new_cluster = clusters - jac / hes
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


def bench_gpu(dataset, k=10, max_iter=10, times=2):
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
        kmeans(max_iter, sp_clusters, sp_features)
        torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        timings[i] = start.elapsed_time(end) * 1000  # micro seconds

    return float(timings[1:].mean()), float(timings[1:].std())
