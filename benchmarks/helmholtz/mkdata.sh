#!/bin/sh

set -e
SEED=0
prec=f32
n=10000

mkdir -p data
futhark dataset --f32-bounds=0.0:$($PYTHON -c "print(1.0/$n)") -b -s $SEED -g f32 -g f32 -g [$n]f32 -g [$n][$n]f32 -g [$n]f32 | gzip > data/n$n.in.gz;

