#!/bin/bash
set -e

# bash ./build.sh

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

QUANT_MODE=$1
BATCH_SIZE=$2

arch=`uname -a | awk -F " " '{print $(NF-1)}'`

if [ ! -d $PROJ_ROOT_PATH/data/images/body25_${QUANT_MODE}_${BATCH_SIZE} ];
then
  mkdir -p $PROJ_ROOT_PATH/data/images/body25_${QUANT_MODE}_${BATCH_SIZE}
fi
if [ ! -d $PROJ_ROOT_PATH/data/images/coco_${QUANT_MODE}_${BATCH_SIZE} ];then
  mkdir -p $PROJ_ROOT_PATH/data/images/coco_${QUANT_MODE}_${BATCH_SIZE}
fi

if [ "$arch" == "x86_64" ];then
    REMOTE_RUN $*
elif [ "$arch" == "aarch64" ];then
    export LD_LIBRARY_PATH=/mps/lib:../../../3rdparty/edge/glog/lib:../../../3rdparty/edge/gflags/lib:../../../3rdparty/edge/opencv/lib:$LD_LIBRARY_PATH
    echo "Begin to inference in ce3226 device."
    ./bin/edge_infer \
        --magicmind_model $MODEL_PATH/pose_body25_${QUANT_MODE}_${BATCH_SIZE}.mm \
        --image_dir $COCO_DATASETS_PATH/val2017/ \
        --output_dir $PROJ_ROOT_PATH/data/images/body25_${QUANT_MODE}_${BATCH_SIZE} \
        --network BODY_25
      
    ./bin/edge_infer \
        --magicmind_model $MODEL_PATH/pose_coco_${QUANT_MODE}_${BATCH_SIZE}.mm \
        --image_dir $COCO_DATASETS_PATH/val2017/ \
        --output_dir $PROJ_ROOT_PATH/data/images/coco_${QUANT_MODE}_${BATCH_SIZE} \
        --network COCO
fi



