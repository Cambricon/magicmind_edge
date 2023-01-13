#!/bin/bash
set -e

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-'1'}

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi

# download datasets and models
cd $PROJ_ROOT_PATH/export_model/
bash run.sh

# gen model 
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $QUANT_MODE $BATCH_SIZE

# build
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh

# clean
rm -rf $PROJ_ROOT_PATH/data/result.json
rm -rf $PROJ_ROOT_PATH/data/output/*

# infer
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh $QUANT_MODE $BATCH_SIZE

# eval
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

# perf
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh $QUANT_MODE $BATCH_SIZE 

# check 
python ${MAGICMIND_EDGE}/utils/check_result.py
