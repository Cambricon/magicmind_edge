#!/bin/bash
set -e

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

QUANT_MODE=$1
BATCH_SIZE=$2

arch=`uname -a | awk -F " " '{print $(NF-1)}'`
if [ "$arch" == "x86_64" ];then
    REMOTE_RUN $*
elif [ "$arch" == "aarch64" ];then
   export LD_LIBRARY_PATH=/mps/lib:../../../3rdparty/edge/glog/lib:../../../3rdparty/edge/gflags/lib:../../../3rdparty/edge/opencv/lib:$LD_LIBRARY_PATH
   echo "Begin to inference in ce3226 device."
   ./bin/edge_infer \
        --magicmind_model ../data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm \
        --image_dir $IJB_DATASETS_PATH/IJBC/loose_crop \
        --image_list $IJB_DATASETS_PATH/IJBC/meta/ijbc_name_5pts_score.txt \
        --save_img true \
        --output_dir $PROJ_ROOT_PATH/data/images
fi
