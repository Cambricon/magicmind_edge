#!/bin/bash
if [ -d  ./bin ];then
  rm -rf ./bin/*
else 
  mkdir bin
fi

echo "Begin to compile host_infer for mlu370 device."
g++ -std=c++11 -O2 -Werror `pkg-config opencv --cflags` -I $THIRD_PARTY/host/ -I $THIRD_PARTY/host/CLI11  -I ./include/ -I $NEUWARE_HOME/include ./src/*.cpp -o ./bin/host_infer -L$NEUWARE_HOME/lib64 -lmagicmind_runtime -lcnrt -lglog `pkg-config opencv --libs` -Werror=unused-result
echo "Compile successed."
