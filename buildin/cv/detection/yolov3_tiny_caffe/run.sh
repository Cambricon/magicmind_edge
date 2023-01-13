#!/bin/bash
set -e

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-'1'}

# gen model 
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $QUANT_MODE $BATCH_SIZE

# build
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh

# clean
rm -rf $PROJ_ROOT_PATH/data/result.json
rm -rf $PROJ_ROOT_PATH/data/images/*

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
