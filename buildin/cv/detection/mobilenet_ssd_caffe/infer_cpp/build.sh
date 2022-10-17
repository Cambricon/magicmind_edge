#!/bin/bash

mkdir -p bin

export NEUWARE_HOME=/usr/local/neuware/edge
export TOOLCHAIN_DIR=/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu

if [ -f "./bin/edge_infer" ];then
    rm ./bin/edge_infer
fi

rm -rf build/*
mkdir -p build
cd build
cmake ..
make -j4
cd ..
