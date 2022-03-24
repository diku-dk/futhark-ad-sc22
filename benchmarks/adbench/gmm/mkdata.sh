#!/bin/sh
#
# Run this to convert ADBench data files to Futhark-compatible data
# files.  They are pretty large, which is why we don't just commit
# them (also, eventually we should get rid of this - the data files
# are massively replicated, and should be constructed on-demand).

set -e

if [ $# -ne 2 ]; then
    echo "Use: $0 path/to/ADBench (f32|f64)"
    exit 1
fi

ADBench=$1
prec=$2
ghc convert.hs
futhark c text2bin.fut

for dir in $(find "$ADBench/data/gmm" -maxdepth 1 -mindepth 1 -type d); do
    mkdir -p data/$(basename $dir)
    for x in $(find $dir -name \*.txt); do
        echo $x
        ./convert < $x | if [ $prec == f32 ]; then sed -r 's/f64/f32/g'; else tee; fi | ./text2bin -b | gzip > data/$(basename $dir)/$(basename -s .txt $x).in.gz;
    done
done
