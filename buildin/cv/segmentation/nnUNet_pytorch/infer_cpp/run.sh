#!/bin/bash
set -e

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-1}
PARAMETER_ID=${3:-1}

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

arch=`uname -a | awk -F " " '{print $(NF-1)}'`
if [ "$arch" == "x86_64" ];then
    REMOTE_RUN $*
elif [ "$arch" == "aarch64" ];then
    export LD_LIBRARY_PATH=/mps/lib:../../../3rdparty/edge/glog/lib:../../../3rdparty/edge/gflags/lib:../../../3rdparty/edge/opencv/lib:$LD_LIBRARY_PATH
    echo "Begin to inference in ce3226 device."
    ./bin/edge_infer \
        --magicmind_model $PROJ_ROOT_PATH/data/models/nnUNet_${QUANT_MODE}_${BATCH_SIZE}_${PARAMETER_ID}.mm \
        --input_data_dir $NNUNET_DATASETS_PATH/nn_UNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
        --softmax_output_dir $PROJ_ROOT_PATH/data/softmax_output \
        --seg_output_dir $PROJ_ROOT_PATH/data/images \
        --batch_size $BATCH_SIZE
fi
