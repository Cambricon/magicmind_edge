#!/bin/bash
set -e

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi

###1.D download caffe model
if [ ! -f "$PROJ_ROOT_PATH/data/models/AlexNet_cvgj/alexnet_cvgj_iter_320000.caffemodel" ];then
    cd $PROJ_ROOT_PATH/data/models
    wget -c https://github.com/cvjena/cnn-models/releases/download/v1.0/cnn-models_cvgj.zip
    unzip -o cnn-models_cvgj.zip
    echo "alexnet's prototxt and caffemodel download in $PROJ_ROOT_PATH/data/models/"
else
    echo "alexnet's prototxt and caffemodel already in $PROJ_ROOT_PATH/data/models/"
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
## bash perf.sh quant_mode batch_size 
bash perf.sh qint8_mixed_float16 1

# check 
python ${MAGICMIND_EDGE}/utils/check_result.py
