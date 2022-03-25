from benchmark import Benchmark, set_precision
from functools import partial
import json
import gzip
import futhark_data
import math
from pathlib import Path

from jax import numpy as jnp, vmap, vjp, tree_map, block_until_ready, jit
from jax.lax import scan
from jax.nn import sigmoid, tanh
from jax.random import normal, split, PRNGKey
import numpy as np
from time import time_ns

data_dir = Path(__file__).parent / "data"


class Helmholtz(Benchmark):
    def __init__(self, n, runs):
        self.runs = runs
        self.n = n
        self.kind = "jax"

    def prepare(self):
        data_file = data_dir / f"n{self.n}.in.gz"
        assert data_file.exists()
        args = tuple(futhark_data.load(gzip.open(data_file)))
        self.args = tuple(map(partial(jnp.array), args))

    def calculate_objective(self, runs):
        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            self.objective = block_until_ready(helmholtz(*self.args))
            timings[i] = (time_ns() - start) / 1000
        return timings

    def calculate_jacobian(self, runs):
        @jit
        def _vjp(*args):
            primals, vjp_fn = vjp(helmholtz, *args)
            return vjp_fn(primals)

        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            jacobian = block_until_ready(_vjp(*self.args))
            timings[i] = (time_ns() - start) / 1000
        return timings

    def validate(self):
        data_file = data_dir / f"{self.name}.out"
        if data_file.exists():
            out = tuple(futhark_data.load(open(data_file)))[0]
            assert np.allclose(
                out, self.objective, rtol=1e-02, atol=1e-02
            )


def bench_all(
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


@jit
def helmholtz(R, T, b, A, xs):
    bxs = jnp.dot(b, xs)
    term1 = sum(jnp.log(xs / (1 - bxs)))
    term2 = jnp.dot(xs, jnp.matmul(A, xs)) / (math.sqrt(8) * bxs)
    term3 = jnp.log((1 + (1 + math.sqrt(2)) * bxs) / (1 + (1 - math.sqrt(2)) * bxs))
    return R * T * term1 - term2 * term3
