#!/bin/bash
set -e

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi

###1.D download caffe model
if [ ! -f "$PROJ_ROOT_PATH/data/models/inception-v3.caffemodel" ];then
    echo "please follow the README.md to download inceptionv3's caffemodel file."
fi
if [ ! -f "$PROJ_ROOT_PATH/data/models/deploy_inception-v3.prototxt" ];then
    echo "please follow the README.md to download inceptionv3's prototxt file."
fi

###2. generate magicmind model#
if [ -f $PROJ_ROOT_PATH/data/models/*.mm ];then
    echo "The mm model exit"
else
    cd $PROJ_ROOT_PATH/gen_model
    ## BUILD_MODEL qint8_mixed_float16  batch_size 
    bash run.sh qint8_mixed_float16 1 
fi

###3. compile the folder: infer_cpp

cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh

###4. infer

if [ -f $PROJ_ROOT_PATH/data/images/* ]; then
    rm $PROJ_ROOT_PATH/data/images/*
fi

cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh qint8_mixed_float16 1

###5. compute accuracy top1/top5
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

###6. benchmark test
cd $PROJ_ROOT_PATH/benchmark
## bash perf.sh quant_mode batch_size threads
bash perf.sh qint8_mixed_float16 1 1
# check 
python ${MAGICMIND_EDGE}/utils/check_result.py
