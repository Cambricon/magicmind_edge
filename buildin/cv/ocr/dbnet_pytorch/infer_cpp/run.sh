#!/bin/bash
set -e
set -x

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

QUANT_MODE=$1  
BATCH_SIZE=$2

if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi
if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${BATCH_SIZE}" ]; 
then
    mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${BATCH_SIZE}"
fi


arch=`uname -a | awk -F " " '{print $(NF-1)}'`
if [ "$arch" == "x86_64" ];then
    REMOTE_RUN $*
elif [ "$arch" == "aarch64" ];then
    export LD_LIBRARY_PATH=/mps/lib:../../../3rdparty/edge/glog/lib:../../../3rdparty/edge/gflags/lib:../../../3rdparty/edge/opencv/lib:$LD_LIBRARY_PATH
    echo "Begin to inference in ce3226 device."
    rm -f $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${BATCH_SIZE}/*
    $PROJ_ROOT_PATH/infer_cpp/bin/edge_infer    --magicmind_model $PROJ_ROOT_PATH/data/models/dbnet_${QUANT_MODE}_${BATCH_SIZE}.mm \
                                --image_dir  $TEXT_DATASETS_PATH/test_images \
                                --save_img False \
                                --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${BATCH_SIZE}

fi