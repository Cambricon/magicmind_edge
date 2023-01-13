#!/bin/bash
if [ -d  ./bin ];then
  rm -rf ./bin/*
else 
  mkdir bin
fi
echo "Begin to compile host_infer for mlu370 device."
g++ -std=c++11 -O2 `pkg-config opencv --cflags` \
    -I $PROJ_ROOT_PATH/infer_cpp \
    -I $THIRD_PARTY/host/ -I $THIRD_PARTY/host/CLI11 \
    -I ./include/ \
    -I $NEUWARE_HOME/include $PROJ_ROOT_PATH/infer_cpp/src/*.cpp -o $PROJ_ROOT_PATH/infer_cpp/bin/host_infer \
    -L$NEUWARE_HOME/lib64 -lmagicmind_runtime -lcnrt \
    -lglog -lgflags `pkg-config opencv --libs`
