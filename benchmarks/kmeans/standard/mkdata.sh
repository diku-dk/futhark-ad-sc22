#!/bin/sh
SEED=0
k=1024
d=2
n=100000000

mkdir -p data
futhark dataset -b -s $SEED -g i32 -g ${k}i64 -g 10i32 -g [$n][$d]f32 | gzip > data/k$k-d$d-n$n.in.gz;
