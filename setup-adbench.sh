#!/bin/sh

set -e

PATH=$PWD/bin/:$PATH

mkdir -p ADBench/build
(cd ADBench/build && cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=OFF ..)
(cd ADBench/build && make -j)
