from benchmark import Benchmark, set_precision
from functools import partial
import json
import gzip
import futhark_data
import math
from pathlib import Path

from jax import numpy as jnp, vmap, vjp, tree_map, block_until_ready, jit, grad
from jax.lax import scan
from jax.nn import sigmoid, tanh
from jax.random import normal, split, PRNGKey
from jax.tree_util import Partial
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
        cs = tuple(futhark_data.load(gzip.open(data_file)))[0:4]
        xs = tuple(futhark_data.load(gzip.open(data_file)))[4]
        self.cs = tuple(map(jnp.array, cs))
        self.xs = jnp.array(xs)

    def calculate_objective(self, runs):
        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            self.objective = block_until_ready(partial(helmholtz, *self.cs)(self.xs))
            timings[i] = (time_ns() - start) / 1000
        return timings

    def calculate_jacobian(self, runs):

        @jit
        def _grad(cs, xs):
          return (grad(Partial(helmholtz, *cs)))(xs)

        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            self.jacobian = block_until_ready(_grad(self.cs, self.xs))
            timings[i] = (time_ns() - start) / 1000
        return timings

    def validate(self):
        obj_file = data_dir / f"n{self.n}.F"
        jac_file = data_dir / f"n{self.n}.J"
        if obj_file.exists():
            obj = tuple(futhark_data.load(open(obj_file)))[0]
            jac = tuple(futhark_data.load(open(jac_file)))
            assert np.allclose(
                obj, self.objective, rtol=1e-02, atol=1e-02
            )
            assert np.allclose(
                   jac, self.jacobian, rtol=1e-02, atol=1e-02
                )

def bench_all(
    ns=[10000], runs=10, output="helmholtz_jax.json", data_dir="data", prec="f32"
):
    set_precision(prec)
    times = {}
    for n in ns:
        helmholtz = Helmholtz(n, runs)
        helmholtz.benchmark()
        times[f"data/n{n}"] = {helmholtz.kind : helmholtz.report()}
    with open(output, "w") as f:
        json.dump(times, f, indent=2)
    print("Benchmarks output to: " + output)
    return


def helmholtz(R, T, b, A, xs):
    bxs = jnp.dot(b, xs)
    term1 = jnp.sum(jnp.log(xs / (1 - bxs)))
    term2 = jnp.dot(xs, jnp.matmul(A, xs)) / (math.sqrt(8) * bxs)
    term3 = jnp.log((1 + (1 + math.sqrt(2)) * bxs) / (1 + (1 - math.sqrt(2)) * bxs))
    return R * T * term1 - term2 * term3
