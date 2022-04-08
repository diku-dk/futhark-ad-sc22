#!/usr/bin/env python3

import json

xsbench_original_file = 'tmp/xsbench-original.txt'
rsbench_original_file = 'tmp/rsbench-original.txt'
lbm_original_file = 'tmp/lbm-original.txt'
xsbench_futhark_file = 'tmp/xsbench-futhark.json'
rsbench_futhark_file = 'tmp/rsbench-futhark.json'
lbm_futhark_file = 'tmp/lbm-futhark.json'

def mean(x):
    return sum(x) / len(x)

xsbench_original_seconds = mean(list(map(float, open(xsbench_original_file).read().split())))
rsbench_original_seconds = mean(list(map(float, open(rsbench_original_file).read().split())))
lbm_original_seconds = mean(list(map(float, open(lbm_original_file).read().split())))

xsbench_futhark_json = json.load(open(xsbench_futhark_file))
rsbench_futhark_json = json.load(open(rsbench_futhark_file))
lbm_futhark_json = json.load(open(lbm_futhark_file))

xsbench_futhark_objective = mean(xsbench_futhark_json['benchmarks/xsbench/xsbench.fut:calculate_objective']['datasets']['data/small.in']['runtimes'])/1e6
xsbench_futhark_jacobian = mean(xsbench_futhark_json['benchmarks/xsbench/xsbench.fut:calculate_jacobian']['datasets']['data/small.in']['runtimes'])/1e6

rsbench_futhark_objective = mean(rsbench_futhark_json['benchmarks/rsbench/rsbench.fut:calculate_objective']['datasets']['data/small.in']['runtimes'])/1e6
rsbench_futhark_jacobian = mean(rsbench_futhark_json['benchmarks/rsbench/rsbench.fut:calculate_jacobian']['datasets']['data/small.in']['runtimes'])/1e6

lbm_futhark_objective = mean(lbm_futhark_json['benchmarks/lbm/lbm.fut:calculate_objective']['datasets']['data/120_120_150_ldc.in']['runtimes'])/1e6
lbm_futhark_jacobian = mean(lbm_futhark_json['benchmarks/lbm/lbm.fut:calculate_jacobian']['datasets']['data/120_120_150_ldc.in']['runtimes'])/1e6

rsbench_futhark_overhead = rsbench_futhark_jacobian / rsbench_futhark_objective
xsbench_futhark_overhead = xsbench_futhark_jacobian / xsbench_futhark_objective
lbm_futhark_overhead = lbm_futhark_jacobian / lbm_futhark_objective

# These are from the Enzyme paper; not recomputed in artifact.
rsbench_enzyme_overhead = 4.2
xsbench_enzyme_overhead = 3.2
lbm_enzyme_overhead = 6.3

print('Benchmark  |    Primal runtimes     |   AD overhead')
print('           | Original      Futhark  | Futhark   Enzyme')
print('-----------+------------------------+----------------------------')
print('RSBench    | %7.3fs      %6.3fs  | %6.1fx  %6.1fx'
      % (rsbench_original_seconds, rsbench_futhark_objective,
         rsbench_futhark_overhead, rsbench_enzyme_overhead))
print('XSBench    | %7.3fs      %6.3fs  | %6.1fx  %6.1fx'
      % (xsbench_original_seconds, xsbench_futhark_objective,
         xsbench_futhark_overhead, xsbench_enzyme_overhead))
print('LBM        | %7.3fs      %6.3fs  | %6.1fx  %6.1fx'
      % (lbm_original_seconds, lbm_futhark_objective,
         lbm_futhark_overhead, lbm_enzyme_overhead))
