#!/bin/bash
set -e

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi

###1. download caffe model
if [ -f $PROJ_ROOT_PATH/data/models/segnet_pascal.prototxt ];then
    echo "segnet_pascal.prototxt already exists."
else
   mkdir -p $PROJ_ROOT_PATH/data/models/
   cd $PROJ_ROOT_PATH/data/models/
   wget -c https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/2d0457ca20a7d22a81f07316bc04b2f26992730c/Example_Models/segnet_pascal.prototxt

fi
if [ -f $PROJ_ROOT_PATH/data/models/segnet_pascal.caffemodel ];then
    echo "segnet_pascal.caffemodel already exists."
else
   mkdir -p $PROJ_ROOT_PATH/data/models/
   cd $PROJ_ROOT_PATH/data/models/
   wget -c http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_pascal.caffemodel 
fi

###2. gen_model#
cd $PROJ_ROOT_PATH/gen_model
bash run.sh qint8_mixed_float16 1 

###2. compile the folder: infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh 

###3. infer
if [ -f $PROJ_ROOT_PATH/data/images/* ] ; 
then
    rm $PROJ_ROOT_PATH/data/images/*
fi

cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh qint8_mixed_float16 1

###4. compute accuracy using coco api
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

###5. benchmark test
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1 

###6. check 
python ${MAGICMIND_EDGE}/utils/check_result.py
