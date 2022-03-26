import json
from collections import namedtuple
from time import time_ns

import jax.random
import numpy as np
from benchmark import Benchmark
from jax import block_until_ready, jit, vmap
from jax import numpy as jnp
from jax import grad
from jax.lax import scan
from jax.nn import sigmoid, tanh
from jax.random import PRNGKey, normal, split
import futhark_data

from lstm_pytorch import gen_filename, parameters

LSTM_WEIGHTS = namedtuple(
    "LSTM_WEIGHTS",
    (
        "w_ii",
        "w_if",
        "w_ig",
        "w_io",
        "w_hi",
        "w_hf",
        "w_hg",
        "w_ho",
        "bi",
        "bf",
        "bg",
        "bo",
        "w_out",
        "b_out",
    ),
)


def read_tensors(filename):
    with open(filename + ".json", "r") as f:
        tensors = json.load(f)
        for name, p in tensors.items():
            tensors[name] = jnp.array(p)
        return tensors


def bench_all(
    runs,
    output,
    parameters=parameters,
    data_dir="data",
    prec="f32",
):
    times = {}
    for params in parameters:
        filename = gen_filename(*params, directory="data")
        tensors = read_tensors(filename)
        lstm = LSTM(tensors, runs, filename)
        lstm.benchmark()
        times[filename] = {lstm.kind: lstm.report()}
    with open(output, "w") as f:
        json.dump(times, f, indent=2)
    return


class LSTM(Benchmark):
    def __init__(self, tensors, runs, filename, hid_dim=5):
        self.runs = runs
        self.kind = "jax"
        self.tensors = tensors
        _, self.run = rnn(hid_dim=hid_dim, num_layers=1)
        self.hid_dim = hid_dim
        self.filename = filename

    def prepare(self):
        self.xs = self.tensors["input"]
        self.target = self.tensors["target"]
        in_weights = tuple(
            jnp.transpose(t) for t in jnp.split(self.tensors["weight_ih_l0"], 4)
        )
        hid_weights = tuple(
            jnp.transpose(t) for t in jnp.split(self.tensors["weight_hh_l0"], 4)
        )
        in_bias = tuple(jnp.split(self.tensors["bias_ih_l0"], 4))
        hid_bias = tuple(jnp.split(self.tensors["bias_hh_l0"], 4))
        bias = tuple(map(sum, zip(in_bias, hid_bias)))
        h_0 = self.tensors["hidn_st0"]
        c_0 = self.tensors["cell_st0"]
        self.weights = [
            LSTM_WEIGHTS(
                *in_weights,
                *hid_weights,
                *bias,
                jnp.transpose(self.tensors["weight"]),
                self.tensors["bias"],
            )
        ]
        self.init_state = jnp.swapaxes(jnp.array([h_0, c_0]), 0, 1)

    def calculate_objective(self, runs):
        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            self.loss = block_until_ready(
                self.run(self.xs, self.init_state, self.weights, self.target)
            )
            timings[i] = (time_ns() - start) / 1000
        return timings

    def calculate_jacobian(self, runs):
        run = lambda weights: self.run(self.xs, self.init_state, weights, self.target)

        @jit
        def _grad(x):
            grad_run = grad(run)
            return grad_run(x)

        timings = np.zeros(runs + 1)
        for i in range(runs + 1):
            start = time_ns()
            self.jacobian = block_until_ready(_grad(self.weights))
            timings[i] = (time_ns() - start) / 1000

        return timings

    def validate(self):
        loss = tuple(futhark_data.load(open(f"{self.filename}.F", "rb")))[0]
        assert np.allclose(loss, self.loss, rtol=1e-02, atol=1e-05)
        # jac = tuple(futhark_data.load(open(f"{self.filename}.J", "rb")))[0]
        # assert np.allclose(jac, self.jacobian, rtol=1e-02, atol=1e-05)


def _lstm_cell(state, weights: LSTM_WEIGHTS, input):
    h, c = state
    i = sigmoid(
        jnp.matmul(input, weights.w_ii) + jnp.matmul(h, weights.w_hi) + weights.bi
    )
    f = sigmoid(
        jnp.matmul(input, weights.w_if) + jnp.matmul(h, weights.w_hf) + weights.bf
    )
    o = sigmoid(
        jnp.matmul(input, weights.w_io) + jnp.matmul(h, weights.w_ho) + weights.bo
    )
    g = tanh(jnp.matmul(input, weights.w_ig) + jnp.matmul(c, weights.w_hg) + weights.bg)
    c = f * c + i * g
    h = o * tanh(c)
    return jnp.stack((h, c)), h


def _vmap_mul(a, b):
    return vmap(lambda alpha: vmap(lambda beta: jnp.sum(alpha * beta))(b.T))(a)


def _lstm_vmap_cell(state, weights: LSTM_WEIGHTS, input):
    h, c = state
    i = sigmoid(
        _vmap_mul(input, weights.w_ii) + _vmap_mul(h, weights.w_hi) + weights.bi
    )
    f = sigmoid(
        _vmap_mul(input, weights.w_if) + _vmap_mul(h, weights.w_hf) + weights.bf
    )
    o = sigmoid(
        _vmap_mul(input, weights.w_io) + _vmap_mul(h, weights.w_ho) + weights.bo
    )
    g = tanh(_vmap_mul(input, weights.w_ig) + _vmap_mul(c, weights.w_hg) + weights.bg)
    c = f * c + i * g
    h = o * tanh(c)
    return jnp.stack((h, c)), h


def _init_lstm_weights(rng_key, in_dim, hid_dim):
    in_key, hid_key = split(rng_key)
    in_weights = normal(in_key, (4, in_dim, hid_dim))
    hid_weights = normal(in_key, (4, hid_dim, hid_dim))
    bias = jnp.zeros((4, hid_dim))
    return LSTM_WEIGHTS(*in_weights, *hid_weights, *bias)


def rnn(hid_dim=5, num_layers=2, lstm_cell=_lstm_vmap_cell):
    def init(rng_seed, in_dim):
        weight_key, state_key = split(rng_seed)
        keys = split(weight_key, num_layers)
        weights = [_init_lstm_weights(keys[0], in_dim, hid_dim)] + [
            _init_lstm_weights(keys[i], hid_dim, hid_dim) for i in range(1, num_layers)
        ]
        # Note: init_state[:, 0] = hs, init_state[:, 1] = cs
        init_state = normal(state_key, (num_layers, 2, hid_dim))
        return init_state, weights

    def _cell(carry, x):
        states, weights = carry
        out_state = []
        h = x
        for i in range(num_layers):
            new_state, h = lstm_cell(states[i], weights[i], h)
            out_state.append(new_state)

        return (jnp.stack(out_state), weights), h

    def run_vmap(xs, init_state, weights, target):
        new_state, y_hat = scan(_cell, (init_state, weights), xs)
        batch_size, steps, _ = y_hat.shape
        jnp.reshape(y_hat, (batch_size * steps, -1))
        y_hat = jnp.matmul(y_hat, weights[-1].w_out) + weights[-1].b_out
        jnp.reshape(y_hat, (batch_size, steps, -1))
        loss = jnp.mean((y_hat - target) ** 2)
        # return new_state, y_hat, loss
        return loss

    return init, run_vmap
