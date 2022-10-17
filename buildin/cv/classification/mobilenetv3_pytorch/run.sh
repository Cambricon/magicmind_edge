#!/bin/bash
set -e
set -x

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi

### 1. download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash get_datasets_and_models.sh

cd $PROJ_ROOT_PATH/export_model
bash run.sh

###2. generate magicmind model#
if [ ! -f $PROJ_ROOT_PATH/data/models/*.mm ];then
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
