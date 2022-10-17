#!/bin/bash
set -e

###0. download datasets and models, export torch.jit.trace pt
cd $PROJ_ROOT_PATH/export_model
#bash run.sh  batch_size paramter_id
bash run.sh 1 0

###1. generate magicmind model#
if [ -f $PROJ_ROOT_PATH/data/models/*.mm ]
then
    echo "The mm model exit"
else
    cd $PROJ_ROOT_PATH/gen_model
    ## BUILD_MODEL qint8_mixed_float16  batch_size  parameter_id
    bash run.sh qint8_mixed_float16 1 0
fi

###2. compile the folder: infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh

###3. infer
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi

if [ -f "$PROJ_ROOT_PATH/data/images/*" ];then
   rm $PROJ_ROOT_PATH/data/images/*
fi

if [ ! -d "$PROJ_ROOT_PATH/data/softmax_output" ];then
    mkdir -p $PROJ_ROOT_PATH/data/softmax_output
fi

if [ -f "$PROJ_ROOT_PATH/data/softmax_output/*" ];then
   rm $PROJ_ROOT_PATH/data/softmax_output/*
fi

cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh quant_model batch_size parameter_id
bash run.sh qint8_mixed_float16 1 0

###4. compute accuracy top1/top5
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

###5. benchmark test
cd $PROJ_ROOT_PATH/benchmark
#bash run.sh quant_model batch_size parameter_id threads
bash perf.sh qint8_mixed_float16 1 0 1

###6. check 
python ${MAGICMIND_EDGE}/utils/check_result.py