#!/bin/bash
set -e

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-'1'}

MODEL_NAME=c3d_${QUANT_MODE}_${BATCH_SIZE}.mm

arch=`uname -a | awk -F " " '{print $(NF-1)}'`
if [ "$arch" == "x86_64" ];then
    REMOTE_RUN $*
elif [ "$arch" == "aarch64" ];then
    # export OPENCV_VIDEOIO_DEBUG=1
    export LD_LIBRARY_PATH=/mps/lib:../../../3rdparty/edge/glog/lib:../../../3rdparty/edge/ffmpeg/lib:../../../3rdparty/edge/gflags/lib:../../../3rdparty/edge/opencv/lib:$LD_LIBRARY_PATH
    echo "Begin to inference in ce3226 device."
    ./bin/edge_infer \
        --magicmind_model $PROJ_ROOT_PATH/data/models/${MODEL_NAME} \
        --video_list $UFC101_DATASETS_PATH/ucfTrainTestlist/testlist01.txt \
        --batch_size $BATCH_SIZE \
        --output_dir $PROJ_ROOT_PATH/data/output \
        --dataset_dir ${UFC101_DATASETS_PATH} \
        --name_file $UFC101_DATASETS_PATH/ucfTrainTestlist/classInd.txt \
        --result_file $PROJ_ROOT_PATH/data/output/infer_result.txt \
        --result_label_file $PROJ_ROOT_PATH/data/output/eval_labels.txt \
        --result_top1_file $PROJ_ROOT_PATH/data/output/eval_result_1.txt \
        --result_top5_file $PROJ_ROOT_PATH/data/output/eval_result_5.txt
fi
