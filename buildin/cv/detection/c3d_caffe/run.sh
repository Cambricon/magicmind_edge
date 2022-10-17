#!/bin/bash
set -e

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-'1'}
THREAD_NUM=${3:-'1'}

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi

###1.D download caffe model
if [ ! -f "$PROJ_ROOT_PATH/data/models/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel" ];then
    cd $PROJ_ROOT_PATH/data/models/
    echo "Downloading  caffemodel file."
    wget https://www.dropbox.com/s/bf5z2jw1pg07c9n/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0 
fi
if [ ! -f "$PROJ_ROOT_PATH/data/models/c3d_resnet18_r2_ucf101.prototxt" ];then
    cd $PROJ_ROOT_PATH/data/models/
    echo "Downloading  prototxt file."
    wget https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/c3d_resnet18_r2_ucf101.prototxt
fi


# gen model 
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $QUANT_MODE $BATCH_SIZE

# build
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh

# clean
rm -rf $PROJ_ROOT_PATH/data/result.json
rm -rf $PROJ_ROOT_PATH/data/output

# infer
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh $QUANT_MODE $BATCH_SIZE

# eval
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

# perf
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh $QUANT_MODE $BATCH_SIZE ${THREAD_NUM}

# check 
python ${MAGICMIND_EDGE}/utils/check_result.py
