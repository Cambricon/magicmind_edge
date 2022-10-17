#!/bin/bash
set -e

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi

if [ ! -d "$PROJ_ROOT_PATH/data/images" ]; then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi

###0. convert torch.jit.trace model
if [ -f $PROJ_ROOT_PATH/data/models/*.pt ];
then
    echo "The torch.jit model exit"
else
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh qint8_mixed_float16 1 
fi

###1. generate magicmind model#
if [ -f "$PROJ_ROOT_PATH/data/models/*.mm" ]; then
    echo "The mm model exit"
else
    cd $PROJ_ROOT_PATH/gen_model
    ## BUILD_MODEL qint8_mixed_float16  batch_size 
    bash run.sh qint8_mixed_float16 1 
fi

###2. compile the folder: infer_cpp

cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh


###3. infer

if [ -f "$PROJ_ROOT_PATH/data/images/*" ]
then
    rm $PROJ_ROOT_PATH/data/images/*
fi

cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh qint8_mixed_float16 1

###4. compute accuracy top1/top5
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

###5. benchmark test
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1 1

###6. check 
python ${MAGICMIND_EDGE}/utils/check_result.py
