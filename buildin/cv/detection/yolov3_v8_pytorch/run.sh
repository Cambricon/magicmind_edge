#!/bin/bash
set -e

pip install -r requirements.txt

###1. convert model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###2. build_model#
cd $PROJ_ROOT_PATH/gen_model
# param: qint8_mixed_float16 batch_size conf iou 
bash run.sh qint8_mixed_float16 1 0.001 0.5 

##2. compile the folder: infer_cpp
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
## bash run.sh quant_mode batch_size  
bash run.sh qint8_mixed_float16 1

###4. compute accuracy using coco api
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

###5. benchmark test
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1 1

###6. check 
python ${MAGICMIND_EDGE}/utils/check_result.py
