from collections import namedtuple

from jax import numpy as jnp, vmap, vjp, tree_map
from jax.lax import scan
from jax.nn import sigmoid, tanh
from jax.random import normal, split, PRNGKey

from benchmark import (Benchmark, set_precision)
import json
from lstm_pytorch import (gen_filename, parameters)

LSTM_WEIGHTS = namedtuple('LSTM_WEIGHTS', ('w_ii', 'w_if', 'w_ig', 'w_io', 'w_hi', 'w_hf', 'w_hg', 'w_ho',
                                           'bi', 'bf', 'bg', 'bo'))

def read_tensors(filename):
  with open(filename + ".json",'r') as f:
   tensors = json.load(f)
   for name, p in tensors.items():
       tensors[name] = jnp.array(p)
   return tensors

def benchmarks(parameters = parameters, runs=10, validate=False, output="lstm_jax.json", data_dir="data", prec="f32"):
  times = {}
  for params in parameters:
    filename = gen_filename(*params, directory="data")
    tensors = read_tensors(filename)
    lstm = LSTM(tensors, runs)
    lstm.benchmark()
    #if validate and not equal(torchLSTM, naiveLSTM):
    #  sys.exit("Error: {filename} does not validate.")
    times[filename] = { lstm.kind : lstm.report(),
                      }
  with open(output,'w') as f:
    json.dump(times, f, indent=2)
  print("Benchmarks output to: " + output)
  return

class LSTM(Benchmark):
    def __init__(self, tensors, runs, hid_dim=5):
        self.runs = runs
        self.kind = "jax"
        self.tensors = tensors
        _, self.run = rnn(hid_dim=hid_dim, num_layers=1)

    def prepare(self):
        self.xs = self.tensors['input']
        self.target = self.tensors['target']
        in_weights = \
             tuple(jnp.transpose(t) for t in jnp.split(self.tensors['weight_ih_l0'], 4))

        hid_weights  = \
             tuple(jnp.transpose(t) for t in jnp.split(self.tensors['weight_hh_l0'], 4))

        in_bias = \
             tuple(jnp.split(self.tensors['bias_ih_l0'], 4))

        hid_bias = \
             tuple(jnp.split(self.tensors['bias_hh_l0'], 4))

        bias = tuple(map(sum, zip(in_bias, hid_bias)))
        h_0 = self.tensors['hidn_st0']
        c_0 = self.tensors['cell_st0']
        self.weights = [LSTM_WEIGHTS(*in_weights, *hid_weights, *bias)]
        self.init_state = jnp.swapaxes(jnp.array([h_0, c_0]), 0, 1)
        self.w_y = jnp.transpose(self.tensors['weight'])
        self.b_y = self.tensors['bias']

    def calculate_objective(self):
       _, self.objective = tree_map(lambda x: x.block_until_ready(), self.run(self.xs, self.init_state, self.weights))

    def calculate_jacobian(self):
       _, self.jacobian = tree_map(lambda x: x.block_until_ready(), vjp(lambda weights: self.run(self.xs, self.init_state, weights), self.weights))

def _lstm_cell(state, weights: LSTM_WEIGHTS, input):
    h, c = state
    i = sigmoid(jnp.matmul(input, weights.w_ii) + jnp.matmul(h, weights.w_hi) + weights.bi)
    f = sigmoid(jnp.matmul(input, weights.w_if) + jnp.matmul(h, weights.w_hf) + weights.bf)
    o = sigmoid(jnp.matmul(input, weights.w_io) + jnp.matmul(h, weights.w_ho) + weights.bo)
    g = tanh(jnp.matmul(input, weights.w_ig) + jnp.matmul(c, weights.w_hg) + weights.bg)
    c = f * c + i * g
    h = o * tanh(c)
    return jnp.stack((h, c)), h


def _init_lstm_weights(rng_key, in_dim, hid_dim):
    in_key, hid_key = split(rng_key)
    in_weights = normal(in_key, (4, in_dim, hid_dim))
    hid_weights = normal(in_key, (4, hid_dim, hid_dim))
    bias = jnp.zeros((4, hid_dim))
    return LSTM_WEIGHTS(*in_weights, *hid_weights, *bias)


def rnn(hid_dim=5, num_layers=2):
    def init(rng_seed, in_dim):
        weight_key, state_key = split(rng_seed)
        keys = split(weight_key, num_layers)
        weights = [_init_lstm_weights(keys[0], in_dim, hid_dim)] + [_init_lstm_weights(keys[i], hid_dim, hid_dim) for i
                                                                    in range(1, num_layers)]
        # Note: init_state[:, 0] = hs, init_state[:, 1] = cs
        init_state = normal(state_key, (num_layers, 2, hid_dim))
        return init_state, weights

    def _cell(carry, x):
        states, weights = carry
        out_state = []
        h = x
        for i in range(num_layers):
            new_state, h = _lstm_cell(states[i], weights[i], h)
            out_state.append(new_state)

        return (jnp.stack(out_state), weights), h

    def run_vmap(xs, init_state, weights):
        #init_state = jnp.repeat(jnp.expand_dims(init_state,2), xs.shape[1], axis=2)
        return scan(_cell, (init_state, weights), xs)

    return init, run_vmap

if __name__ == '__main__':
    rng_seed = PRNGKey(43)
    hid_dim = 5
    in_dim = 2
    num_layers = 1
    lengths = 4
    num_datum = 6
    data_seed, init_seed = split(rng_seed)

    xs = normal(data_seed, (lengths, num_datum, in_dim))  # time-major

    init, run = rnn(hid_dim=hid_dim, num_layers=num_layers)

    init_state, weights = init(rng_seed=rng_seed, in_dim=in_dim)
    states, out = run(xs, init_state, weights)

    primals, run_vjp = vjp(lambda weights: run(xs, init_state, weights), weights)
    grads = run_vjp(primals)[0][0]
    print(grads.w_ii)
