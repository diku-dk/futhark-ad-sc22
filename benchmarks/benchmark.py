from abc import ABC, abstractmethod
import torch
import json
import os
import numpy as np
from time import time_ns

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

  def time_fun(self, f):
    if self.kind is "pytorch" or None:
      timings = np.zeros((self.runs +1 ,))
      start = torch.cuda.Event(enable_timing=True)
      end   = torch.cuda.Event(enable_timing=True)
      start.record()
      for i in range(self.runs + 1):
        start.record()
        f()
        torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        timings[i] = start.elapsed_time(end)*1000
    elif self.kind is "jax":
        timings = f(self.runs)
    return float(timings[1:].mean()), float(timings[1:].std())

  def time_objective(self):
      self.objective_time, self.objective_std = self.time_fun(self.calculate_objective)
      print(self.objective_time)

  def time_jacobian(self):
      self.jacobian_time, self.jacobian_std = self.time_fun(self.calculate_jacobian)
      print(self.jacobian_time)

  def benchmark(self):
    self.prepare()
    self.time_objective()
    self.time_jacobian()

  def report(self):
    return { 'objective': self.objective_time,
             'objective_std': self.objective_std,
             'jacobian': self.jacobian_time,
             'jacobian_std': self.jacobian_std,
             'overhead': self.jacobian_time/self.objective_time
           }

def set_precision(prec):
  if (prec == "f32"):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_dtype(torch.float32)
  elif (prec == "f64"):
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    torch.set_default_dtype(torch.float64)
  else:
    sys.exit("Error: invalid precision " + prec)

def process_futhark(path, source, name='futhark'):
  with open(path,'r') as f:
    fut = json.load(f)
    objective = fut[source+':calculate_objective']['datasets']
    res = {}
    for d in objective.keys():
        objs = objective[d]['runtimes']
        obj_time = sum(objs)/len(objs)
        (d_, _) = os.path.splitext(d)
        if len(list(fut.values())) > 1:
          jacobian = fut[source+':calculate_jacobian']['datasets']
          jacs = jacobian[d]['runtimes']
          jac_time = sum(jacs)/len(jacs)
          res[d_] = ({ name : { 'objective' : obj_time,
                              'jacobian'  : jac_time,
                              'overhead'  : jac_time/obj_time
                            }
                       })
        else:
          res[d_] = ({ name : { 'objective' : obj_time}
                     })

  with open(path,'w') as f:
    json.dump(res, f, sort_keys=True, indent=2)
    
def process(paths, jac_speedup = True, obj_speedup = False):
  res = {}
  for p in paths:
    with open(p, 'r') as f:
      j = json.load(f)
      for (d,r) in j.items():
        if d not in res:
          res[d] = r
        else:
          res[d].update(r)
  for d, v in res.items():
    for l in v.keys():
        for z in v.keys():
            if z is not l:
              if jac_speedup:
                v[z]['jac_speedup_' + l] = v[l]['jacobian'] / v[z]['jacobian']
              if obj_speedup:
                v[z]['obj_speedup_' + l] = v[l]['objective'] / v[z]['objective']
  return res

def get_results(d = "./"):
  res = []
  for file in os.listdir(d):
      if file.endswith(".json"):
          res.append(os.path.join(d, file))
  return res

def dump(out_path, paths = get_results(), jac_speedup = True, obj_speedup = False):
  d = process(paths, jac_speedup, obj_speedup)
  with open(out_path,'w') as f:
    json.dump(d, f, sort_keys=True, indent=2)

def pretty(paths):
  d = process(paths)
  print(json.dumps(d, sort_keys=True, indent=2))

def ms(n):
  return round(n/1000, 1)

def r(n):
  return round(n, 1)

def latex_gmm(paths):
  print(paths)
  d = process(paths)
  d0 = d['data/1k/gmm_d64_K200']
  d1 = d['data/1k/gmm_d128_K200']
  d2 = d['data/10k/gmm_d32_K200']
  d3 = d['data/10k/gmm_d64_K25']
  d4 = d['data/10k/gmm_d128_K25']
  d5 = d['data/10k/gmm_d128_K200']
  print(f"""
                                      & $\\mathbf{{D}}_0$                        & $\\mathbf{{D}}_1$                       & $\\mathbf{{D}}_2$                       & $\\mathbf{{D}}_3$                       & $\\mathbf{{D}}_4$                       & $\\mathbf{{D}}_5$\\\\ \\hline
\\textbf{{PyT. Jacob. (ms)}}          & ${ms(d0['pytorch']['jacobian'])}$        & ${ms(d1['pytorch']['jacobian'])}$       & ${ms(d2['pytorch']['jacobian'])}$       & ${ms(d3['pytorch']['jacobian'])}$       & ${ms(d4['pytorch']['jacobian'])}$       & ${ms(d5['pytorch']['jacobian'])}$       \\\\
\\textbf{{Fut. Speedup ($\\times$)}}  & ${r(d0['futhark']['jac_speedup_pytorch'])}$  & ${r(d1['futhark']['jac_speedup_pytorch'])}$ & ${r(d2['futhark']['jac_speedup_pytorch'])}$ & ${r(d3['futhark']['jac_speedup_pytorch'])}$ & ${r(d4['futhark']['jac_speedup_pytorch'])}$ & ${r(d5['futhark']['jac_speedup_pytorch'])}$ \\\\
\\textbf{{PyT. Overhead ($\\times$)}} & ${r(d0['pytorch']['overhead'])}$         & ${r(d1['pytorch']['overhead'])}$        & ${r(d2['pytorch']['overhead'])}$        & ${r(d3['pytorch']['overhead'])}$        & ${r(d4['pytorch']['overhead'])}$        & ${r(d5['pytorch']['overhead'])}$        \\\\
\\textbf{{Fut. Overhead ($\\times$)}} & ${r(d0['pytorch']['overhead'])}$         & ${r(d1['pytorch']['overhead'])}$        & ${r(d2['pytorch']['overhead'])}$        & ${r(d3['pytorch']['overhead'])}$        & ${r(d4['pytorch']['overhead'])}$        & ${r(d5['pytorch']['overhead'])}$         
""")

def latex_lstm(paths):
  d = process(paths)
  d0 = d['data/lstm-bs1024-n20-d300-h192']
  d1 = d['data/lstm-bs1024-n300-d80-h256']
  print(f"""
         \\multirow{{2}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\scriptsize\\textbf{{gpu}}}}}}   & \\multicolumn{{1}}{{|c}}{{$\\mathbf{{D_0}}$}} & \\multicolumn{{1}}{{c|}}{{${ms(d0['naive']['jacobian'])}$}}  &  {r(d0['futhark']['speedup_naive'])} & \\multicolumn{{1}}{{c|}}{{{r(d0['futhark']['speedup_naive'])}}} & {r(d0['naive']['overhead'])} & {r(d0['futhark']['overhead'])}  & {r(d0['torch.nn.LSTM']['overhead'])} \\\\
                                                                                              & \\multicolumn{{1}}{{|c}}{{$\\mathbf{{D_1}}$}} & \\multicolumn{{1}}{{c|}}{{${ms(d1['naive']['jacobian'])}$}}  &  {r(d1['futhark']['speedup_naive'])} & \\multicolumn{{1}}{{c|}}{{{r(d1['futhark']['speedup_naive'])}}} & {r(d1['naive']['overhead'])} & {r(d1['futhark']['overhead'])}  & {r(d1['torch.nn.LSTM']['overhead'])} \\\\\\hline
         """)

def latex_kmeans(paths):
  d = process(paths, False, True)
  d0 = d['data/kdd_cup']
  d1 = d['data/random']
  print(f"""
  $(k,n,d)$          & \\textbf{{Manual}}                & \\multicolumn{{1}}{{c|}}{{\\textbf{{AD}}}} & \\textbf{{PyTorch}}                 & \\textbf{{JAX}} \\\\\\hline
  $(5,494019,35)$    & ${ms(d0['manual']['objective'])}$ & ${ms(d0['futhark']['objective'])}$         & ${ms(d0['pytorch']['objective'])}$ & ${ms(d0['jax']['objective'])}$ \\\\
  $(1024,10000,256)$ & ${ms(d1['manual']['objective'])}$ & ${ms(d1['futhark']['objective'])}$         & ${ms(d1['pytorch']['objective'])}$ & ${ms(d1['jax']['objective'])}$ \\\\ \\hline
  """)
