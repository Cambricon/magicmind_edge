#!/bin/bash
set -e

###1. download caffe model
if [ -f $PROJ_ROOT_PATH/data/models/deploy.prototxt ];then
    cd $PROJ_ROOT_PATH/data/models
    if grep -q "refinedet_param {" deploy.prototxt;
    then
      echo "modifying file: deploy.prototxt has been already done"
    else
      echo "modifying file: deploy.prototxt ..."
      patch -u deploy.prototxt < deploy.diff
    fi
else
   cd $PROJ_ROOT_PATH/data/models/
   echo "Please download by yourself: https://drive.google.com/file/d/1d1T_tTImZynD88CoB0OF0rdFXgp4E_le/view"
   echo "Follow these steps :"
   echo "       tar -xvf models_VGGNet_VOC0712Plus_refinedet_vgg16_320x320.tar.gz"
   echo "       mv models/VGGNet/VOC0712Plus/refinedet_vgg16_320x320/deploy.prototxt models/VGGNet/VOC0712Plus/refinedet_vgg16_320x320/VOC0712Plus_refinedet_vgg16_320x320_final.caffemodel ./  "
   echo "       bash run.sh again"
   exit 1 
fi

if [ -f $PROJ_ROOT_PATH/data/models/VOC0712Plus_refinedet_vgg16_320x320_final.caffemodel ];then
    echo "caffemodel already exists."
else
   cd $PROJ_ROOT_PATH/data/models/
   echo "caffemodel doesn't exists,please check README."
   exit 1 
fi

###2. build_model#
if [ -f $PROJ_ROOT_PATH/data/models/refinedet_qint8_mixed_float16_1.mm ];then
    echo "refinedet.mm model already exists."
else 
    cd $PROJ_ROOT_PATH/gen_model
    ## BUILD_MODEL qint8_mixed_float16 batch_size 
    bash run.sh qint8_mixed_float16 1 
fi

###2. compile the folder: infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh 

###3. infer
if [ -f $PROJ_ROOT_PATH/data/images/* ]; 
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
