#!/bin/sh

set -e
SEED=0
prec=f32

mkdir -p data
for n in $(seq 1 50); do
    futhark dataset -b -s $SEED -g f32 -g f32 -g [$n]f32 -g [$n][$n]f32 -g [$n]f32 | gzip > data/n$n.in.gz;
done

