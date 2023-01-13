#!/bin/bash
set -e

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

QUANT_MODE=$1
BATCH_SIZE=$2
if [ ! -d $PROJ_ROOT_PATH/data/outout ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi

if [ -f $PROJ_ROOT_PATH/data/output/tusimple_eval_tmp.0.txt ]; 
then
    rm -rf $PROJ_ROOT_PATH/data/output/*
fi

arch=`uname -a | awk -F " " '{print $(NF-1)}'`
if [ "$arch" == "x86_64" ];then
    REMOTE_RUN $*
elif [ "$arch" == "aarch64" ];then
    export LD_LIBRARY_PATH=/mps/lib:../../../3rdparty/edge/glog/lib:../../../3rdparty/edge/gflags/lib:../../../3rdparty/edge/opencv/lib:$LD_LIBRARY_PATH
    echo "Begin to inference in ce3226 device."
    ./bin/edge_infer \
        --magicmind_model $PROJ_ROOT_PATH/data/models/tusimple_${QUANT_MODE}_${BATCH_SIZE}.mm \
        --image_dir $TUSIMPLE_DATASETS_PATH/ \
        --output_dir $PROJ_ROOT_PATH/data/output \
        --file_list $TUSIMPLE_DATASETS_PATH/test.txt 
fi
