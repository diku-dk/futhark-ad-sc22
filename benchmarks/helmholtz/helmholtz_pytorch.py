from benchmark import Benchmark, set_precision
from functools import partial
import json
import gzip
import futhark_data
import torch
import math

from pathlib import Path
from torch.autograd.functional import vhp
from torch.autograd.functional import vjp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_dir = Path(__file__).parent / "data"


class Helmholtz(Benchmark):
    def __init__(self, n, runs):
        self.runs = runs
        self.n = n
        self.kind = "pytorch"

    def prepare(self):
        data_file = data_dir / f"n{self.n}.in.gz"
        assert data_file.exists()
        args = tuple(futhark_data.load(gzip.open(data_file)))
        self.args = tuple(map(partial(torch.tensor), args))

    def calculate_objective(self):
        self.objective = helmholtz(*self.args)

    def calculate_jacobian(self):
        self.jacobian = vjp(helmholtz, self.args)


def benchmarks(
    ns=[50], runs=10, output="helmholtz_pytorch.json", data_dir="data", prec="f32"
):
    set_precision(prec)
    times = {}
    for n in ns:
        helmholtz = Helmholtz(n, runs)
        helmholtz.benchmark()
        times[f"data/n{n}"] = {"pytorch": helmholtz.report()}
    with open(output, "w") as f:
        json.dump(times, f, indent=2)
    print("Benchmarks output to: " + output)
    return


def helmholtz(R, T, b, A, xs):
    bxs = torch.dot(b, xs)
    term1 = sum(torch.log(xs / (1 - bxs)))
    term2 = torch.dot(xs, torch.matmul(A, xs)) / (math.sqrt(8) * bxs)
    term3 = torch.log((1 + (1 + math.sqrt(2)) * bxs) / (1 + (1 - math.sqrt(2)) * bxs))
    return R * T * term1 - term2 * term3
