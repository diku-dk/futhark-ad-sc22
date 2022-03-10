import json
import os

def process(py_path, fut_path):
  with open(py_path,'r') as py, open(fut_path,'r') as fut:
    res = json.load(py)
    fut = json.load(fut)
    objective = list(fut.values())[0]['datasets']
    jacobian = list(fut.values())[1]['datasets']
    for d in objective.keys():
        objs = objective[d]['runtimes']
        jacs = jacobian[d]['runtimes']
        obj_time = sum(objs)/len(objs)
        jac_time = sum(jacs)/len(jacs)
        (d_, _) = os.path.splitext(d)
        res[d_].update({ 'futhark' : { 'objective' : obj_time,
                                       'jacobian'  : jac_time,
                                       'overhead'  : jac_time/obj_time
                                     }
                      })
    for d, v in res.items():
      for l in v.keys():
          for z in v.keys():
              if z is not l:
                  v[z]['speedup_' + l] = v[l]['jacobian'] / v[z]['jacobian']
    return res

def dump(py_path, fut_path, out_path):
  d = process(py_path, fut_path)
  with open(out_path,'w') as f:
    json.dump(d, f, sort_keys=True, indent=2)

def pretty(py_path, fut_path):
  d = process(py_path, fut_path)
  print(json.dumps(d, sort_keys=True, indent=2))

def ms(n):
  return round(n/1000, 1)

def r(n):
  return round(n, 1)

def latex(py_path, fut_path, out_path):
  d = process(py_path, fut_path)
  d0 = d['data/lstm-bs1024-n20-d300-h192']
  d1 = d['data/lstm-bs1024-n300-d80-h256']
  with open(out_path,'w') as f:
    f.write(
        f"""
         \\multirow{{2}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\scriptsize\\textbf{{gpu}}}}}}   & \\multicolumn{{1}}{{|c}}{{$\\mathbf{{D_0}}$}} & \\multicolumn{{1}}{{c|}}{{${ms(d0['naive']['jacobian'])}$}}  &  {r(d0['futhark']['speedup_naive'])} & \\multicolumn{{1}}{{c|}}{{{r(d0['futhark']['speedup_naive'])}}} & {r(d0['naive']['overhead'])} & {r(d0['futhark']['overhead'])}  & {r(d0['torch.nn.LSTM']['overhead'])} \\\\
                                                                                              & \\multicolumn{{1}}{{|c}}{{$\\mathbf{{D_1}}$}} & \\multicolumn{{1}}{{c|}}{{${ms(d1['naive']['jacobian'])}$}}  &  {r(d1['futhark']['speedup_naive'])} & \\multicolumn{{1}}{{c|}}{{{r(d1['futhark']['speedup_naive'])}}} & {r(d1['naive']['overhead'])} & {r(d1['futhark']['overhead'])}  & {r(d1['torch.nn.LSTM']['overhead'])} \\\\\\hline
         """)
