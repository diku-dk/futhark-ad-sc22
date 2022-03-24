import gzip
import json
from functools import partial
from pathlib import Path

import futhark_data
import numpy as np
import torch
from benchmark import Benchmark, set_precision
from torch.autograd.functional import vhp, vjp

data_dir = Path(__file__).parent / "data"


class KMeans(Benchmark):
    def __init__(self, name, runs, device):
        self.runs = runs
        self.name = name
        self.device = device
        self.kind = "pytorch"

    def prepare(self):
        self.args = data_gen(self.name, self.device)

    def calculate_objective(self):
        self.iterations, self.objective = kmeans(*self.args)

    def calculate_jacobian(self):
        return

    def validate(self):
        data_file = data_dir / f"{self.name}.out"
        if data_file.exists():
            t, out = tuple(futhark_data.load(open(data_file, "rb")))
            print(self.iterations)
            print(t)
            #assert (self.iterations + 1 == t)
            assert np.allclose(
                out, self.objective.cpu().detach().numpy(), rtol=1e-02, atol=1e-05
            )


def bench_all(runs, output, datasets=["kdd_cup", "random"], prec="f32"):
    set_precision(prec)
    times = {}
    for data in datasets:
        kmeans = KMeans(data, runs, "cuda")
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
    a_sqr = (a**2).sum(1)[None, :]
    b_sqr = (b**2).sum(1)[:, None]
    diff = torch.matmul(b, a.T)
    return a_sqr + b_sqr - 2 * diff


def cost(points, centers):
    dists = all_pairs_norm(points, centers)
    (min_dist, _) = torch.min(dists, dim=0)
    return min_dist.sum()


def kmeans(_, k, max_iter, features, _tolerance=1):
    tolerance = 1.05
    clusters = torch.flip(features[-int(k) :], (0,))
    t = 0
    converged = False
    while not converged and t < max_iter:
        _, jac = vjp(partial(cost, features), clusters, v=torch.tensor(1.0))
        _, hes = vhp(partial(cost, features), clusters, v=torch.ones_like(clusters))

        new_cluster = clusters - jac / hes
        converged = ((new_cluster - clusters) ** 2).sum() < tolerance
        clusters = new_cluster
        t += 1
    return t, clusters


def data_gen(name, device):
    data_file = data_dir / f"{name}.in.gz"
    assert data_file.exists()
    kmeans_args = tuple(futhark_data.load(gzip.open(data_file)))
    return tuple(
        map(partial(torch.tensor, device=device, dtype=torch.float32), kmeans_args)
    )
