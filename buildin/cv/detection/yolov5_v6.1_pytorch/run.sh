#!/bin/bash
set -e

###1. convert model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###2. build_model#
cd $PROJ_ROOT_PATH/gen_model
# BUILD_MODEL quant_mode batch_size conf iou 
bash run.sh qint8_mixed_float16 1 0.001 0.6 

###2. compile the folder: infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh 

###3. infer
if [ -f $PROJ_ROOT_PATH/data/result.json ]; 
then
    rm $PROJ_ROOT_PATH/data/result.json
fi

if [ -f $PROJ_ROOT_PATH/data/images/* ]; 
then
    rm $PROJ_ROOT_PATH/data/images/*
fi

cd $PROJ_ROOT_PATH/infer_cpp
# bash run.sh quant_mode batch_size  
bash run.sh qint8_mixed_float16 1

###4. compute accuracy using coco api
bash $PROJ_ROOT_PATH/benchmark/eval.sh 

###5. benchmark test
cd $PROJ_ROOT_PATH/benchmark
## bash perf.sh quant_mode batch_size threads
bash perf.sh qint8_mixed_float16 1 1

###6. check 
python ${MAGICMIND_EDGE}/utils/check_result.py
