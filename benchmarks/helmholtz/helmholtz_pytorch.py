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
import numpy as np

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
        cs = tuple(futhark_data.load(gzip.open(data_file)))[0:4]
        xs = tuple(futhark_data.load(gzip.open(data_file)))[4]
        self.cs = tuple(map(lambda a: torch.tensor(a, requires_grad=False), cs))
        self.xs = torch.tensor(xs, requires_grad=True)

    def calculate_objective(self):
        self.objective = partial(helmholtz, *self.cs)(self.xs)

    def calculate_jacobian(self):
        self.jacobian = vjp(partial(helmholtz, *self.cs), self.xs)

    def validate(self):
        obj_file = data_dir / f"n{self.n}.F"
        jac_file = data_dir / f"n{self.n}.J"
        if obj_file.exists():
            obj = tuple(futhark_data.load(open(obj_file)))[0]
            jac = tuple(futhark_data.load(open(jac_file)))
            assert np.allclose(
                obj, self.objective.cpu().detach().numpy(), rtol=1e-02, atol=1e-02
            )
            assert np.allclose(
                   jac, self.jacobian[1].cpu().detach().numpy(), rtol=1e-02, atol=1e-02
                )


def bench_all(
    ns=[10000], runs=10, output="helmholtz_pytorch.json", data_dir="data", prec="f32"
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
