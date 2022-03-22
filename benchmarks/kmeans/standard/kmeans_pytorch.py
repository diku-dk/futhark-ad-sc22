import argparse
from functools import partial
from pathlib import Path

import futhark_data
import torch
from torch.autograd.functional import vhp
from torch.autograd.functional import vjp

from benchmark import (Benchmark, set_precision)
import json
import gzip

data_dir = Path(__file__).parent / 'data'

class KMeans(Benchmark):
    def __init__(self, name, runs, device):
        self.runs = runs
        self.name = name
        self.device = device
        self.kind = "pytorch"

    def prepare(self):
        self.args = data_gen(self.name, self.device)

    def calculate_objective(self):
       self.objective = kmeans(*self.args)

    def calculate_jacobian(self):
        return

def benchmarks(datasets = ["kdd_cup", "random"], runs=10, output="kmeans_pytorch.json"):
  set_precision("f32")
  times = {}
  for data in datasets:
    kmeans = KMeans(data, runs, "cuda")
    kmeans.benchmark()
    times['data/' + data] = { 'pytorch' : 
                           { 'objective': kmeans.objective_time,
                             'objective_std': kmeans.objective_std,
                           }
                  }
  with open(output,'w') as f:
    json.dump(times, f, indent=2)
  print("Benchmarks output to: " + output)
  return

def all_pairs_norm(a, b):
    a_sqr = (a ** 2).sum(1)[None, :]
    b_sqr = (b ** 2).sum(1)[:, None]
    diff = torch.matmul(b, a.T)
    return a_sqr + b_sqr - 2 * diff


def cost(points, centers):
    dists = all_pairs_norm(points, centers)
    (min_dist,_) = torch.min(dists, dim=0)
    return min_dist.sum()


def kmeans(_, k, max_iter, features, tolerance=1):

    clusters = torch.flip(features[-int(k):], (0,))
    t = 0
    converged = False
    while not converged and t < max_iter:
        _, jac = vjp(partial(cost, features), clusters, v=torch.tensor(1.))
        _, hes = vhp(partial(cost, features), clusters, v=torch.ones_like(clusters))

        new_cluster = clusters - jac / hes
        converged = ((new_cluster - clusters) ** 2).sum() < tolerance
        clusters = new_cluster
        t += 1
    return clusters


def bench(kmeans_args, times=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = torch.zeros((times,), dtype=float)
    for i in range(times):
        torch.cuda.synchronize()
        start.record()
        kmeans(*kmeans_args)
        torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        timings[i] = start.elapsed_time(end) * 1000  # micro seconds

    return float(timings[1:].mean()), float(timings[1:].std())


def data_gen(name, device):
    data_file = data_dir / f'{name}.in.gz'
    assert data_file.exists()
    kmeans_args = tuple(futhark_data.load(gzip.open(data_file)))
    return tuple(map(partial(torch.tensor, device=device, dtype=torch.float32), kmeans_args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Newtonian KMeans"
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--datasets", nargs="+", default=["kdd_cup", "random"])

    args = parser.parse_args()
    if args.device == 'cuda':
        assert torch.cuda.is_available(), "Cuda not available"

    for name in args.datasets:
        print(name, '±'.join(map(str, bench(data_gen(name, args.device)))), 'micro seconds')
