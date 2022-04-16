import json
import os
import re
from abc import ABC, abstractmethod
from prettytable import PrettyTable

import numpy as np
import torch


class Benchmark(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def prepare(self):
        ...

    @abstractmethod
    def calculate_objective(self):
        ...

    @abstractmethod
    def calculate_jacobian(self):
        ...

    @abstractmethod
    def validate(self):
        ...

    def time_fun(self, f):
        if self.kind == "pytorch":
            timings = np.zeros((self.runs + 1,))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i in range(self.runs + 1):
                start.record()
                f()
                torch.cuda.synchronize()
                end.record()
                torch.cuda.synchronize()
                timings[i] = start.elapsed_time(end) * 1000
        elif self.kind == "jax" or self.kind == "jax-vmap":
            timings = f(self.runs)
        return float(timings[1:].mean()), float(timings[1:].std())

    def time_objective(self):
        self.objective_time, self.objective_std = self.time_fun(
            self.calculate_objective
        )

    def time_jacobian(self):
        self.jacobian_time, self.jacobian_std = self.time_fun(self.calculate_jacobian)

    def benchmark(self):
        self.prepare()
        self.time_objective()
        self.time_jacobian()
        self.validate()

    def report(self):
        return {
            "objective": self.objective_time,
            "objective_std": self.objective_std,
            "jacobian": self.jacobian_time,
            "jacobian_std": self.jacobian_std,
            "overhead": self.jacobian_time / self.objective_time,
        }


def set_precision(prec):
    if prec == "f32":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.set_default_dtype(torch.float32)
    elif prec == "f64":
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        torch.set_default_dtype(torch.float64)
    else:
        sys.exit("Error: invalid precision " + prec)

def union_dict(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))

def process_futhark(name, path, source, basename):
    name = "futhark" if name == basename else re.sub("^.*_", "", basename)
    with open(path, "r") as f:
        fut = json.load(f)
        if source + ":calculate_objective2" in fut:
          objective = union_dict(fut[source + ":calculate_objective"]["datasets"], fut[source + ":calculate_objective2"]["datasets"])
        else:
          objective = fut[source + ":calculate_objective"]["datasets"]
        res = {}
        for d in objective.keys():
            objs = objective[d]["runtimes"]
            obj_time = sum(objs) / len(objs)
            (d_, _) = os.path.splitext(d)
            if source + ":calculate_jacobian" in fut:
                jacobian = fut[source + ":calculate_jacobian"]["datasets"]
                jacs = jacobian[d]["runtimes"]
                jac_time = sum(jacs) / len(jacs)
                res[d_] = {
                    name: {
                        "objective": obj_time,
                        "jacobian": jac_time,
                        "overhead": jac_time / obj_time,
                    }
                }
            else:
                res[d_] = {name: {"objective": obj_time}}

    with open(path, "w") as f:
        json.dump(res, f, sort_keys=True, indent=2)


def process(paths, jac_speedup, obj_speedup):
    res = {}
    for p in paths:
        with open(p, "r") as f:
            j = json.load(f)
            for (d, r) in j.items():
                if d not in res:
                    res[d] = r
                else:
                    res[d].update(r)
    for d, v in res.items():
        for l in v.keys():
            for z in v.keys():
                if z is not l:
                    if jac_speedup:
                        v[z]["jac_speedup_" + l] = v[l]["jacobian"] / v[z]["jacobian"]
                    if obj_speedup:
                        v[z]["obj_speedup_" + l] = v[l]["objective"] / v[z]["objective"]
    return res


def get_results(d="./"):
    res = []
    for file in os.listdir(d):
        if file.endswith(".json") and "result" not in file:
            res.append(os.path.join(d, file))
    return res


def dump(out_path, jac_speedup, obj_speedup, paths=get_results()):
    d = process(paths, jac_speedup, obj_speedup)
    with open(out_path, "w") as f:
        json.dump(d, f, sort_keys=True, indent=2)


def pretty(paths):
    d = process(paths)
    print(json.dumps(d, sort_keys=True, indent=2))


def ms(n):
    return f"{round(n / 1e3, 1)}ms"

def ms_(n):
    return round(n / 1e3, 1)

def s(n, d=2):
    return f"{round(n / 1e6, d):.2f}s"


def r(n):
    return round(n, 1)

def t(n):
    return f"{round(n, 1)}\\times"

def t_(n):
    return f"{round(n, 1)}x"

def ml(f, d, k1, k2):
    try:
        return f'{f(d[k1][k2])}'
    except KeyError:
        return "-"


def table(name, res):
    tab = PrettyTable()
    d = json.load(open(res))
    if name == "gmm":
        d0 = d["data/f64/1k/gmm_d64_K200"]
        d1 = d["data/f64/1k/gmm_d128_K200"]
        d2 = d["data/f64/10k/gmm_d32_K200"]
        d3 = d["data/f64/10k/gmm_d64_K25"]
        d4 = d["data/f64/10k/gmm_d128_K25"]
        d5 = d["data/f64/10k/gmm_d128_K200"]
        print('Measurement       |   D0    D1    D2    D3    D4   D5')
        print('------------------+------------------------------------')
        print('Pyt. Jacob. (ms)  |%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f'
              % (ms_(d0['pytorch']['jacobian']),
                 ms_(d1['pytorch']['jacobian']),
                 ms_(d2['pytorch']['jacobian']),
                 ms_(d3['pytorch']['jacobian']),
                 ms_(d4['pytorch']['jacobian']),
                 ms_(d5['pytorch']['jacobian'])))
        print('Fut. Speedup (x)  |%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f'
              % (r(d0['futhark']['jac_speedup_pytorch']),
                 r(d1['futhark']['jac_speedup_pytorch']),
                 r(d2['futhark']['jac_speedup_pytorch']),
                 r(d3['futhark']['jac_speedup_pytorch']),
                 r(d4['futhark']['jac_speedup_pytorch']),
                 r(d5['futhark']['jac_speedup_pytorch'])))
        print('PyT. Overhead (x) |%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f'
              % (r(d0['pytorch']['overhead']),
                 r(d1['pytorch']['overhead']),
                 r(d2['pytorch']['overhead']),
                 r(d3['pytorch']['overhead']),
                 r(d4['pytorch']['overhead']),
                 r(d5['pytorch']['overhead'])))
        print('Fut. Overhead (x) |%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f'
              % (r(d0['futhark']['overhead']),
                 r(d1['futhark']['overhead']),
                 r(d2['futhark']['overhead']),
                 r(d3['futhark']['overhead']),
                 r(d4['futhark']['overhead']),
                 r(d5['futhark']['overhead'])))

    elif name == "lstm":
        d0 = d["data/lstm-bs1024-n20-d300-h192"]
        d1 = d["data/lstm-bs1024-n300-d80-h256"]
        print('   |               |              Speedups')
        print('   |PyTorch Jacob. |  Futhark   nn.LSTM       JAX  JAX(VMap)')
        print('---+---------------+----------------------------------------')
        print('D0 |%14s |%9s %9s %9s %10s'
              % (ms(d0['pytorch']['jacobian']),
                 t_(d0['futhark']['jac_speedup_pytorch']),
                 t_(d0['torch.nn.LSTM']['jac_speedup_pytorch']),
                 ml(t_,d0,'jax','jac_speedup_pytorch'),
                 ml(t_,d0,'jax-vmap','jac_speedup_pytorch')))
        print('D1 |%14s |%9s %9s %9s %10s'
              % (ms(d1['pytorch']['jacobian']),
                 t_(d1['futhark']['jac_speedup_pytorch']),
                 t_(d1['torch.nn.LSTM']['jac_speedup_pytorch']),
                 ml(t_,d1,'jax','jac_speedup_pytorch'),
                 ml(t_,d1,'jax-vmap','jac_speedup_pytorch')))
        print('')
        print('')
        print('   |                      Overheads')
        print('   |  PyTorch   Futhark   nn.LSTM       JAX  JAX(VMap)')
        print('---+--------------------------------------------------')
        print('D0 |%9s %9s %9s %9s %9s'
              % (t_(d0['pytorch']['overhead']),
                 t_(d0['futhark']['overhead']),
                 t_(d0['torch.nn.LSTM']['overhead']),
                 ml(t_,d0,'jax','overhead'),
                 ml(t_,d0,'jax-vmap','overhead')))
        print('D1 |%9s %9s %9s %9s %9s'
              % (t_(d1['pytorch']['overhead']),
                 t_(d1['futhark']['overhead']),
                 t_(d1['torch.nn.LSTM']['overhead']),
                 ml(t_,d1,'jax','overhead'),
                 ml(t_,d1,'jax-vmap','overhead')))


    elif name == "kmeans":
        d0 = d["data/kdd_cup"]
        d1 = d["data/random"]
        d2 = d["data/k1024-d10-n2000000"]
        print('Data |    Futhark      | Pytorch       JAX   JAX(VMap)')
        print('     | Manual      AD  |')
        print('-----+-----------------+------------------------------')
        print('D0   |%7s  %7s |%8s  %8s %11s'
              % 
          (ms(d0['manual']['objective']),
           ms(d0['futhark']['objective']),
           ms(d0['pytorch']['objective']),
           ml(ms, d0, 'jax', 'objective'),
           ml(ms, d0, 'jax-vmap', 'objective')))
        print('D1   |%7s  %7s |%8s  %8s %11s'
              % 
          (ms(d1['manual']['objective']),
           ms(d1['futhark']['objective']),
           ms(d1['pytorch']['objective']),
           ml(ms, d1, 'jax', 'objective'),
           ml(ms, d1, 'jax-vmap', 'objective')))
        print('D2   |%7s  %7s |%8s  %8s %11s'
              % 
          (ms(d2['manual']['objective']),
           ms(d2['futhark']['objective']),
           ms(d2['pytorch']['objective']),
           ml(ms, d2, 'jax', 'objective'),
           ml(ms, d2, 'jax-vmap', 'objective')))

    elif name == "kmeans_sparse":
        d0 = d["data/movielens"]
        d1 = d["data/nytimes"]
        d2 = d["data/scrna"]
        print('          |    Futhark      |')
        print('Workload  | Manual       AD | Pytorch      JAX')
        print('----------+-----------------+-----------------')
        print('movielens |%7s  %7s | %7s  %7s'
              % (s(d0['manual']['objective']),
                 s(d0['futhark']['objective']),
                 s(d0['pytorch']['objective']),
                 ml(s,d0,'jax','objective')))
        print('  nytimes |%7s  %7s | %7s  %7s'
              % (s(d1['manual']['objective']),
                 s(d1['futhark']['objective']),
                 s(d1['pytorch']['objective']),
                 ml(s,d1,'jax','objective')))
        print('    scrna |%7s  %7s | %7s  %7s'
              % (s(d2['manual']['objective']),
                 s(d2['futhark']['objective']),
                 s(d2['pytorch']['objective']),
                 ml(s,d2,'jax','objective')))

def latex(name, jac_speedup, obj_speedup, res):
    d = json.load(open(res))
    if name == "gmm":
        d0 = d["data/f64/1k/gmm_d64_K200"]
        d1 = d["data/f64/1k/gmm_d128_K200"]
        d2 = d["data/f64/10k/gmm_d32_K200"]
        d3 = d["data/f64/10k/gmm_d64_K25"]
        d4 = d["data/f64/10k/gmm_d128_K25"]
        d5 = d["data/f64/10k/gmm_d128_K200"]
        print(
            f"""
                                          & $\\mathbf{{D}}_0$                        & $\\mathbf{{D}}_1$                       & $\\mathbf{{D}}_2$                       & $\\mathbf{{D}}_3$                       & $\\mathbf{{D}}_4$                       & $\\mathbf{{D}}_5$\\\\ \\hline
    \\textbf{{PyT. Jacob. (ms)}}          & ${ms_(d0['pytorch']['jacobian'])}$        & ${ms_(d1['pytorch']['jacobian'])}$       & ${ms_(d2['pytorch']['jacobian'])}$       & ${ms_(d3['pytorch']['jacobian'])}$       & ${ms_(d4['pytorch']['jacobian'])}$       & ${ms_(d5['pytorch']['jacobian'])}$       \\\\
    \\textbf{{Fut. Speedup ($\\times$)}}  & ${r(d0['futhark']['jac_speedup_pytorch'])}$  & ${r(d1['futhark']['jac_speedup_pytorch'])}$ & ${r(d2['futhark']['jac_speedup_pytorch'])}$ & ${r(d3['futhark']['jac_speedup_pytorch'])}$ & ${r(d4['futhark']['jac_speedup_pytorch'])}$ & ${r(d5['futhark']['jac_speedup_pytorch'])}$ \\\\
    \\textbf{{PyT. Overhead ($\\times$)}} & ${r(d0['pytorch']['overhead'])}$         & ${r(d1['pytorch']['overhead'])}$        & ${r(d2['pytorch']['overhead'])}$        & ${r(d3['pytorch']['overhead'])}$        & ${r(d4['pytorch']['overhead'])}$        & ${r(d5['pytorch']['overhead'])}$        \\\\
    \\textbf{{Fut. Overhead ($\\times$)}} & ${r(d0['futhark']['overhead'])}$         & ${r(d1['futhark']['overhead'])}$        & ${r(d2['futhark']['overhead'])}$        & ${r(d3['futhark']['overhead'])}$        & ${r(d4['futhark']['overhead'])}$        & ${r(d5['futhark']['overhead'])}$"""
        )

    elif name == "lstm":
        d0 = d["data/lstm-bs1024-n20-d300-h192"]
        d1 = d["data/lstm-bs1024-n300-d80-h256"]
        print(
            f"""
           \\multicolumn{{1}}{{c|}}{{$\\mathbf{{D}}_0$}} & \\multicolumn{{1}}{{c|}}{{${ms(d0['pytorch']['jacobian'])}$}}  & ${t(d0['futhark']['jac_speedup_pytorch'])}$ & ${t(d0['torch.nn.LSTM']['jac_speedup_pytorch'])}$ & ${ml(t,d0,'jax','jac_speedup_pytorch')}$ & ${ml(t,d0,'jax-vmap','jac_speedup_pytorch')}$ \\\\ 
           \\multicolumn{{1}}{{c|}}{{$\\mathbf{{D}}_1$}} & \\multicolumn{{1}}{{c|}}{{${ms(d1['pytorch']['jacobian'])}$}}  & ${t(d1['futhark']['jac_speedup_pytorch'])}$ & ${t(d1['torch.nn.LSTM']['jac_speedup_pytorch'])}$ & ${ml(t,d1,'jax','jac_speedup_pytorch')}$ & ${ml(t,d1,'jax-vmap','jac_speedup_pytorch')}$ \\\\""")
        print(
            f"""
           \\multicolumn{{1}}{{c|}}{{$\\mathbf{{D}}_0$}} & ${t(d0['pytorch']['overhead'])}$ & ${t(d0['futhark']['overhead'])}$  & ${t(d0['torch.nn.LSTM']['overhead'])}$ & ${ml(t,d0,'jax','overhead')}$ & ${ml(t,d0,'jax-vmap','overhead')}$\\\\
           \\multicolumn{{1}}{{c|}}{{$\\mathbf{{D}}_1$}} & ${t(d1['pytorch']['overhead'])}$ & ${t(d1['futhark']['overhead'])}$  & ${t(d1['torch.nn.LSTM']['overhead'])}$ & ${ml(t,d1,'jax','overhead')}$ & ${ml(t,d1,'jax-vmap','overhead')}$\\\\\\hline"""
        )

    elif name == "kmeans":
        d0 = d["data/kdd_cup"]
        d1 = d["data/random"]
        d2 = d["data/k1024-d10-n2000000"]
        print(
            f"""
  & \\multicolumn{{1}}{{c|}}{{$\\mathbf{{D}}_0$}}   & ${ms(d0['manual']['objective'])}$ & ${ms(d0['futhark']['objective'])}$   & ${ms(d0['pytorch']['objective'])}$ & ${ml(ms, d0, 'jax', 'objective')}$ & ${ml(ms, d0, 'jax-vmap', 'objective')}$\\\\
  & \\multicolumn{{1}}{{c|}}{{$\\mathbf{{D}}_1$}}   & ${ms(d1['manual']['objective'])}$ & ${ms(d1['futhark']['objective'])}$   & ${ms(d1['pytorch']['objective'])}$ & ${ml(ms, d1, 'jax', 'objective')}$ & ${ml(ms, d1, 'jax-vmap', 'objective')}$\\\\
  & \\multicolumn{{1}}{{c|}}{{$\\mathbf{{D}}_2$}}   & ${ms(d2['manual']['objective'])}$ & ${ms(d2['futhark']['objective'])}$   & ${ms(d2['pytorch']['objective'])}$ & ${ml(ms, d2, 'jax', 'objective')}$ & ${ml(ms, d2, 'jax-vmap', 'objective')}$\\\\ \\hline
    """
        )

    elif name == "kmeans_sparse":
            d0 = d["data/movielens"]
            d1 = d["data/nytimes"]
            d2 = d["data/scrna"]
            print(
                f"""
                & movielens & ${s(d0['manual']['objective'])}$ & ${s(d0['futhark']['objective'])}$ & ${s(d0['pytorch']['objective'])}$ & ${ml(s,d0,'jax','objective')}$\\\\
                & nytimes   & ${s(d1['manual']['objective'])}$ & ${s(d1['futhark']['objective'])}$ & ${s(d1['pytorch']['objective'])}$ & ${ml(s,d1,'jax','objective')}$\\\\
                & scrna     & ${s(d2['manual']['objective'])}$ & ${s(d2['futhark']['objective'])}$ & ${s(d2['pytorch']['objective'])}$ & ${ml(s,d2,'jax','objective')}$\\\\
                """)
    

    elif name == "rsbench":
        d0 = d["data/small"]
        print(
            f"\\textbf{{RSBench}} & $2.311s$ & ${s(d0['futhark']['objective'])}$ & ${r(d0['futhark']['overhead'])}\\times$ & $4.2\\times$ \\\\"
        )

    elif name == "xsbench":
        d0 = d["data/small"]
        print(
            f"\\textbf{{XSBench}} & $0.244s$ & ${s(d0['futhark']['objective'])}$ & ${r(d0['futhark']['overhead'])}\\times$ & $3.2\\times$ \\\\"
        )

    elif name == "lbm":
        d0 = d["data/120_120_150_ldc"]
        print(
            f"\\textbf{{LBM}} & $0.071s$ & ${s(d0['futhark']['objective'])}$ & ${r(d0['futhark']['overhead'])}\\times$ & $6.3\\times$ \\\\"
        )
