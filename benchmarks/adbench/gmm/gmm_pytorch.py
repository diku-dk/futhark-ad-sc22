# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import math
import torch
import futhark_data
import gzip
from benchmark import (Benchmark, set_precision)
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

datasets = [ "data/1k/gmm_d64_K200"
           , "data/1k/gmm_d128_K200"
           , "data/10k/gmm_d32_K200"
           , "data/10k/gmm_d64_K25"
           , "data/10k/gmm_d128_K25"
           , "data/10k/gmm_d128_K200"
           ]

def benchmarks(paths = datasets, runs=10, output="gmm_pytorch.json", prec="f64"):
  set_precision(prec)
  times = {}
  for path in paths:
     g = futhark_data.load(gzip.open(path + ".in.gz"))
     gmm = PyTorchGMM(runs, list(g))
     gmm.benchmark()
     times[path] = { 'pytorch': gmm.report() }
  with open(output,'w') as f:
    json.dump(times, f, indent=2)
  print("Benchmarks output to: " + output)
  return

class PyTorchGMM(torch.nn.Module, Benchmark):
    def __init__(self, runs, inputs):
        super().__init__()
        self.runs = runs
        self.inputs = to_torch_tensors(
            (inputs[0], inputs[1], inputs[2]), grad_req = True
        )

        self.params = to_torch_tensors(
            (inputs[3], inputs[4], inputs[5])
        )
        self.kind = "pytorch"

    def prepare(self):
        self.objective = torch.zeros(1, device=device)
        self.gradient = torch.empty(0, device=device)

    def output(self):
        return self.objective.item(), self.gradient.numpy()

    def calculate_objective(self):
        self.objective = gmm_objective(*self.inputs, *self.params)

    def calculate_jacobian(self):
        self.objective, self.gradient = torch_jacobian(gmm_objective, self.inputs, self.params)


def log_wishart_prior(p, wishart_gamma, wishart_m, sum_qs, Qdiags, icf):
    n = p + wishart_m + 1
    k = icf.shape[0]

    out = torch.sum(
        0.5 * wishart_gamma * wishart_gamma *
        (torch.sum(Qdiags ** 2, dim=1) + torch.sum(icf[:, p:] ** 2, dim=1)) -
        wishart_m * sum_qs
    )

    C = n * p * (math.log(wishart_gamma / math.sqrt(2)))
    return out - k * (C - torch.special.multigammaln(.5 * n, p))


def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    n = x.shape[0]
    d = x.shape[1]

    Qdiags = torch.exp(icf[:, :d])
    sum_qs = torch.sum(icf[:, :d], 1)

    to_from_idx = torch.nn.functional.pad(torch.cumsum(torch.arange(d - 1, 0, -1), 0) + d, (1, 0),
                                          value=d) - torch.arange(1, d + 1)
    idx = torch.tril(torch.arange(d).expand((d, d)).T + to_from_idx[None, :], -1)
    Ls = icf[:, idx] * (idx > 0)[None, ...]

    xcentered = x[:, None, :] - means[None, ...]

    Lxcentered = Qdiags * xcentered + torch.einsum('ijk,mik->mij', Ls, xcentered)
    sqsum_Lxcentered = torch.sum(Lxcentered ** 2, 2)
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    lse = torch.logsumexp(inner_term, 1)
    slse = torch.sum(lse)

    CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return CONSTANT + slse - n * torch.logsumexp(alphas, 0) \
           + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)


def gmm_jacobian(inputs, params):
    torch_jacobian(gmm_objective, inputs = inputs, params = params)

def to_torch_tensor(param, grad_req = False, dtype = torch.float64):
    '''Converts given single parameter to torch tensors. Note that parameter
    can be an ndarray-like object.
    
    Args:
        param (ndarray-like): parameter to convert.
        grad_req (bool, optional): defines flag for calculating tensor
            jacobian for created torch tensor. Defaults to False.
        dtype (type, optional): defines a type of tensor elements. Defaults to
            torch.float64.

    Returns:
        torch tensor
    '''

    return torch.tensor(
        param,
        dtype = dtype,
        requires_grad = grad_req,
        device=device
    )



def to_torch_tensors(params, grad_req = False, dtype = torch.float64):
    '''Converts given multiple parameters to torch tensors. Note that
    parameters can be ndarray-lake objects.
    
    Args:
        params (enumerable of ndarray-like): parameters to convert.
        grad_req (bool, optional): defines flag for calculating tensor
            jacobian for created torch tensors. Defaults to False.
        dtype (type, optional): defines a type of tensor elements. Defaults to
            torch.float64.

    Returns:
        tuple of torch tensors
    '''

    return tuple(
        torch.tensor(param, dtype = dtype, requires_grad = grad_req, device=device)
        for param in params
    )



def torch_jacobian(func, inputs, params = None, flatten = True):
    '''Calculates jacobian and return value of the given function that uses
    torch tensors.

    Args:
        func (callable): function which jacobian is calculating.
        inputs (tuple of torch tensors): function inputs by which it is
            differentiated.
        params (tuple of torch tensors, optional): function inputs by which it
            is doesn't differentiated. Defaults to None.
        flatten (bool, optional): if True then jacobian will be written in
            1D array row-major. Defaults to True.

    Returns:
        torch tensor, torch tensor: function result and function jacobian.
    '''

    def recurse_backwards(output, inputs, J, flatten):
        '''Recursively calls .backward on multi-dimensional output.'''

        def get_grad(tensor, flatten):
            '''Returns tensor gradient flatten representation. Added for
            performing concatenation of scalar tensors gradients.'''

            if tensor.dim() > 0:
                if flatten:
                    return tensor.grad.flatten()
                else:
                    return tensor.grad
            else:
                return tensor.grad.view(1)


        if output.dim() > 0:
            for item in output:
                recurse_backwards(item, inputs, J, flatten)
        else:
            for inp in inputs:
                inp.grad = None

            output.backward(retain_graph = True)

            J.append(torch.cat(
                list(get_grad(inp, flatten) for inp in inputs)
            ))

    if params != None:
        res = func(*inputs, *params)
    else:
        res = func(*inputs)

    J = []
    recurse_backwards(res, inputs, J, flatten)

    J = torch.stack(J)

    if flatten:
        J = J.t().flatten()
    return res, J
