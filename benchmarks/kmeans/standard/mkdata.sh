#!/bin/sh
SEED=0
k=1024
d=10
n1=2000000
n2=3000000

mkdir -p data
futhark dataset -b -s $SEED -g i32 -g ${k}i64 -g 10i32 -g [$n1][$d]f32 | gzip > data/k$k-d$d-n$n1.in.gz;
futhark dataset -b -s $SEED -g i32 -g ${k}i64 -g 10i32 -g [$n2][$d]f32 | gzip > data/k$k-d$d-n$n2.in.gz;
