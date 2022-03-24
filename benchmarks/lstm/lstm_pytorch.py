import json
import os
from itertools import chain

import futhark_data
import torch
import torch.nn as nn
from benchmark import Benchmark, set_precision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

parameters = [(1024, 20, 300, 192), (1024, 300, 80, 256)]


def equal(m1, m2):
    jacobians_equal = True
    for n in m1.jacobian.keys():
        jacobians_equal = jacobians_equal and torch.allclose(
            m1.jacobian[n], m2.jacobian[n], 1e-03, 1e-05
        )
    return (
        torch.allclose(m1.objective, m2.objective, 1e-03, 1e-03)
        and torch.allclose(m1.loss, m2.loss, 1e-03, 1e-03)
        #and jacobians_equal
    )


def gen_filename(bs, n, d, h, directory, ext=None):
    path = f"{directory}/lstm-bs{bs}-n{n}-d{d}-h{h}"
    return path if ext is None else f"{path}.{ext}"


def read_tensors(filename):
    with open(filename + ".json", "r") as f:
        tensors = json.load(f)
        for name, p in tensors.items():
            tensors[name] = torch.tensor(p)
        return tensors


def gen_data(parameters=parameters, data_dir="data", prec="f32"):
    set_precision(prec)
    torch.manual_seed(0)
    for params in parameters:
        filename = gen_filename(*params, data_dir)
        torchLSTM = RNNLSTM(*params, filename, tensors=None, runs=None)
        torchLSTM.gen_data()
    return


def bench_all(runs, output, parameters=parameters, data_dir="data", prec="f32"):
    set_precision(prec)
    times = {}
    for params in parameters:
        filename = gen_filename(*params, directory="data")
        tensors = read_tensors(filename)
        torchLSTM = RNNLSTM(*params, filename, tensors, runs)
        naiveLSTM = NaiveLSTM(tensors, runs)
        torchLSTM.benchmark()
        naiveLSTM.benchmark()
        assert equal(torchLSTM, naiveLSTM)
        times[filename] = {
            "pytorch": naiveLSTM.report(),
            "torch.nn.LSTM": torchLSTM.report(),
        }
    with open(output, "w") as f:
        json.dump(times, f, indent=2)
    return


class NaiveLSTM(nn.Module, Benchmark):
    def __init__(
        self,
        tensors,
        runs,
        activation_h=nn.Tanh,
        activation_o=nn.Sigmoid,
        activation_f=nn.Sigmoid,
        activation_i=nn.Sigmoid,
        activation_j=nn.Tanh,
    ):
        super().__init__()

        # it is fine to hard code these
        self.activation_h = activation_h()
        self.activation_o = activation_o()
        self.activation_f = activation_f()
        self.activation_i = activation_i()
        self.activation_j = activation_j()
        self.tensors = tensors
        self.runs = runs
        self.kind = "pytorch"

    def prepare(self):
        self.input_ = self.tensors["input"]
        self.target = self.tensors["target"]
        # parameters of the (recurrent) hidden layer
        self.W_i, self.W_f, self.W_j, self.W_o = tuple(
            nn.Parameter(torch.transpose(t, 0, 1))
            for t in torch.chunk(self.tensors["weight_ih_l0"], 4)
        )

        self.b_ii, self.b_if, self.b_ij, self.b_io = tuple(
            nn.Parameter(t) for t in torch.chunk(self.tensors["bias_ih_l0"], 4)
        )

        self.b_hi, self.b_hf, self.b_hj, self.b_ho = tuple(
            nn.Parameter(t) for t in torch.chunk(self.tensors["bias_hh_l0"], 4)
        )

        self.U_i, self.U_f, self.U_j, self.U_o = tuple(
            nn.Parameter(torch.transpose(t, 0, 1))
            for t in torch.chunk(self.tensors["weight_hh_l0"], 4)
        )

        # initial hidden state
        self.h_0 = nn.Parameter(self.tensors["hidn_st0"][0])
        self.c_0 = nn.Parameter(self.tensors["cell_st0"][0])

        # output layer (fully connected)
        self.W_y = nn.Parameter(torch.transpose(self.tensors["weight"], 0, 1))
        self.b_y = nn.Parameter(self.tensors["bias"])

        self.input_ = nn.Parameter(torch.transpose(self.tensors["input"], 0, 1))
        self.target = nn.Parameter(torch.transpose(self.tensors["target"], 0, 1))

    def step(self, x_t, h, c):
        #  forward pass for a single time step
        j = self.activation_j(
            torch.matmul(x_t, self.W_j)
            + torch.matmul(h, self.U_j)
            + self.b_ij
            + self.b_hj
        )
        i = self.activation_i(
            torch.matmul(x_t, self.W_i)
            + torch.matmul(h, self.U_i)
            + self.b_ii
            + self.b_hi
        )
        f = self.activation_f(
            torch.matmul(x_t, self.W_f)
            + torch.matmul(h, self.U_f)
            + self.b_if
            + self.b_hf
        )
        o = self.activation_o(
            torch.matmul(x_t, self.W_o)
            + torch.matmul(h, self.U_o)
            + self.b_io
            + self.b_ho
        )

        c = f * c + i * j

        h = o * self.activation_h(c)

        return h, c  # returning new hidden and cell state

    def iterate_series(self, x, h, c):
        # apply rnn to each time step and give an output (many-to-many task)
        batch_size, n_steps, dimensions = x.shape

        # can use cell states list here but only the last cell is required
        hidden_states = []
        # iterate over time axis (1)
        for t in range(n_steps):
            # give previous hidden state and input from the current time step
            h, c = self.step(x[:, t], h, c)
            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states, 1)

        # fully connected output
        y_hat = hidden_states.reshape(
            batch_size * n_steps, -1
        )  # flatten steps and batch size (bs * )
        y_hat = torch.matmul(y_hat, self.W_y) + self.b_y
        y_hat = y_hat.reshape(batch_size, n_steps, -1)  # regains structure
        return y_hat, hidden_states[:, -1], c

    def calculate_objective(self):
        y_hat, h, c = self.iterate_series(self.input_, self.h_0, self.c_0)
        self.objective = torch.transpose(y_hat, 0, 1)
        self.loss = torch.mean((y_hat - self.target) ** 2)
        return y_hat, h, c

    def calculate_jacobian(self):
        self.zero_grad()
        # get predictions (forward pass)
        y_hat, h, c = self.calculate_objective()

        #self.loss = torch.mean((y_hat - self.target) ** 2)
        # backprop
        self.loss.backward(gradient=torch.tensor(1.0))

        d = {n: p.grad for n, p in self.named_parameters()}
        self.jacobian = {
            "weight_ih_l0": torch.concat(
                [
                    torch.transpose(g, 0, 1)
                    for g in [d["W_i"], d["W_f"], d["W_j"], d["W_o"]]
                ]
            ),
            "weight_hh_l0": torch.concat(
                [
                    torch.transpose(g, 0, 1)
                    for g in [d["U_i"], d["U_f"], d["U_j"], d["U_o"]]
                ]
            ),
            "bias_ih_l0": torch.concat([d["b_ii"], d["b_if"], d["b_ij"], d["b_io"]]),
            "bias_hh_l0": torch.concat([d["b_hi"], d["b_hf"], d["b_hj"], d["b_ho"]]),
            "weight": torch.transpose(d["W_y"], 0, 1),
            "bias": d["b_y"],
        }
        return

    def validate(self):
        return  # we validate above


class RNNLSTM(nn.Module, Benchmark):
    def __init__(self, bs, n, h, d, filename, tensors, runs):

        super(RNNLSTM, self).__init__()
        self.num_layers = 1
        self.bs = bs
        self.n = n
        self.h = h
        self.d = d
        self.lstm = nn.LSTM(
            input_size=self.d,
            hidden_size=self.h,
            num_layers=self.num_layers,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
            proj_size=0,
        )
        self.linear = nn.Linear(self.h, self.d)
        self.filename = filename
        self.tensors = tensors
        self.runs = runs
        self.kind = "pytorch"

    def prepare(self):
        if self.tensors is None:
            self.inputs = torch.randn(self.n, self.bs, self.d).to(device)
            self.target = torch.randn(self.n, self.bs, self.d).to(device)
            self.hidn_st0 = torch.zeros(self.num_layers, self.bs, self.h).to(device)
            self.cell_st0 = torch.zeros(self.num_layers, self.bs, self.h).to(device)
        else:
            self.inputs = self.tensors["input"].to(device)
            self.target = self.tensors["target"].to(device)
            self.hidn_st0 = self.tensors["hidn_st0"].to(device)
            self.cell_st0 = self.tensors["cell_st0"].to(device)
            with torch.no_grad():
                for n, p in chain(
                    self.lstm.named_parameters(), self.linear.named_parameters()
                ):
                    p.copy_(self.tensors[n].clone().detach())

    def forward(
        self,
        input_,
    ):
        outputs, st = self.lstm(input_, (self.hidn_st0, self.cell_st0))
        self.objective = torch.reshape(
            self.linear(torch.cat([t for t in outputs])), (self.n, self.bs, self.d)
        )
        loss_function = nn.MSELoss(reduction="mean")
        self.loss = loss_function(self.objective, self.target)
        return self.objective

    def calculate_objective(self):
        self.forward(self.inputs)

    def calculate_jacobian(self):
        self.zero_grad()
        self.forward(self.inputs)
        #loss_function = nn.MSELoss(reduction="mean")
        #self.loss = loss_function(self.objective, self.target)
        self.loss.backward(gradient=torch.tensor(1.0))
        self.jacobian = {
            n: p.grad
            for n, p in chain(
                self.lstm.named_parameters(), self.linear.named_parameters()
            )
        }

    def validate(self):
        return  # we validate above

    def dump(self):
        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))
        with open(self.filename + ".F", "wb") as f:
            futhark_data.dump(self.loss.cpu().detach().numpy(), f, True)
        with open(self.filename + ".J", "wb") as f:
            for n, g in self.jacobian.items():
                if n == "weight":
                    futhark_data.dump(g.cpu().detach().numpy().T, f, True)
                else:
                    futhark_data.dump(g.cpu().detach().numpy(), f, True)

    def dump_inputs(self):
        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))
        d = {
            "input": self.inputs,
            "target": self.target,
            "hidn_st0": self.hidn_st0,
            "cell_st0": self.cell_st0,
        }
        for name, p in chain(
            self.lstm.named_parameters(), self.linear.named_parameters()
        ):
            d[name] = p

        d_futhark = {}
        for name, p in d.items():
            xs = p.cpu().detach().numpy()
            if name == "hidn_st0":
                d_futhark[name] = xs[0, :, :].T
            elif name == "cell_st0":
                d_futhark[name] = xs[0, :, :].T
            elif name == "weight":
                d_futhark[name] = xs.T
            else:
                d_futhark[name] = xs

        d_futhark["loss_adj"] = torch.tensor(1.0).cpu().detach().numpy()

        with open(self.filename + ".json", "w") as f:
            json.dump({name: p.tolist() for name, p in d.items()}, f)

        with open(self.filename + ".in", "wb") as f:
            for xs in d_futhark.values():
                futhark_data.dump(xs, f, True)

    def gen_data(self):
        self.prepare()
        self.calculate_objective()
        self.calculate_jacobian()
        self.dump_inputs()
        self.dump()
