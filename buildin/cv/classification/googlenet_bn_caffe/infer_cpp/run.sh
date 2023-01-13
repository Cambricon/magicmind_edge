#!/bin/bash
set -e
set -x
QUANT_MODE=$1
BATCH_SIZE=$2

source ${MAGICMIND_EDGE}/utils/remote_tools.sh
arch=`uname -a | awk -F " " '{print $(NF-1)}'`
if [ "$arch" == "x86_64" ];then
    REMOTE_RUN $*
elif [ "$arch" == "aarch64" ];then
    echo "Begin to inference in ce3226 device."
    export LD_LIBRARY_PATH=/mps/lib:../../../3rdparty/edge/glog/lib:../../../3rdparty/edge/gflags/lib:../../../3rdparty/edge/opencv/lib:$LD_LIBRARY_PATH
    echo "Begin to inference in ce3226 device."
    $PROJ_ROOT_PATH/infer_cpp/bin/edge_infer   \
        --magicmind_model $PROJ_ROOT_PATH/data/models/googlenet_bn_${QUANT_MODE}_${BATCH_SIZE}.mm \
        --image_dir $IMAGENET_DATASETS_PATH \
        --output_dir $PROJ_ROOT_PATH/data/images \
        --batch_size ${BATCH_SIZE}
fi
