#!/bin/bash
set -e

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi
###1. download caffemodel
if [ ! -f "$PROJ_ROOT_PATH/data/models/VGG_ILSVRC_16_layers.caffemodel" ]; then
    echo "Downloading caffemodel..."
    cd $PROJ_ROOT_PATH/data/models/
    wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
fi

if [ ! -f "$PROJ_ROOT_PATH/data/models/VGG_ILSVRC_16_layers_deploy.prototxt" ]; then
    echo "Downloading prototxt..."
    cd $PROJ_ROOT_PATH/data/models/
    wget -c https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt
fi

###2. generate magicmind model#

cd $PROJ_ROOT_PATH/gen_model
## BUILD_MODEL qint8_mixed_float16  batch_size 
bash run.sh qint8_mixed_float16 1 

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
