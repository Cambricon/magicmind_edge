#!/bin/bash
set -e

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-'1'}

MODEL_NAME=yolov3_tiny_caffe_${QUANT_MODE}_${BATCH_SIZE}.mm

arch=`uname -a | awk -F " " '{print $(NF-1)}'`
if [ "$arch" == "x86_64" ];then
    REMOTE_RUN $*
elif [ "$arch" == "aarch64" ];then
    export LD_LIBRARY_PATH=/mps/lib:../../../3rdparty/edge/glog/lib:../../../3rdparty/edge/gflags/lib:../../../3rdparty/edge/opencv/lib:$LD_LIBRARY_PATH
    echo "Begin to inference in ce3226 device."
    mkdir -p $PROJ_ROOT_PATH/data/images
    ./bin/edge_infer \
        --magicmind_model $PROJ_ROOT_PATH/data/models/${MODEL_NAME} \
        --image_dir $COCO_DATASETS_PATH/val2017/ \
        --output_dir $PROJ_ROOT_PATH/data/images \
        --save_img true \
        --label_path ${UTILS_PATH}/coco.names \
        --coco_result "../data/result.json"
fi
