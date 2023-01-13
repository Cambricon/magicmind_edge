#!/bin/bash
set -e
set -x

###0. convert model
pip install -r requirement.txt
cd $PROJ_ROOT_PATH/export_model 
bash run.sh

###1. build_model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision batch_size
bash run.sh qint8_mixed_float16 1
###2. compile the folder: infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh

if [ -f $PROJ_ROOT_PATH/data/images/* ]; 
then
    rm $PROJ_ROOT_PATH/data/images/*
fi
###3. infer
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh quant_mode batch_size
bash run.sh qint8_mixed_float16 1

###4. compute accuracy using coco api
bash $PROJ_ROOT_PATH/benchmark/eval.sh qint8_mixed_float16 1

###5. benchmark test
cd $PROJ_ROOT_PATH/benchmark
## bash perf.sh quant_mode batch_size threads
bash perf.sh qint8_mixed_float16 1 1

###6. check 
python ${MAGICMIND_EDGE}/utils/check_result.py