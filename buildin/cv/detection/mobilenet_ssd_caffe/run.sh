#!/bin/bash
set -e

###1. download caffe model
if [ -f $PROJ_ROOT_PATH/data/models/deploy.prototxt ];then
    echo "deploy.prototxt already exists."
else
   cd $PROJ_ROOT_PATH/data/models/
   wget -c https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/deploy.prototxt --no-check-certificate 

fi
if [ -f $PROJ_ROOT_PATH/data/models/mobilenet_iter_73000.caffemodel ];then
    echo "caffemodel already exists."
else
   cd $PROJ_ROOT_PATH/data/models/
   wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel
fi

###2. build_model#
if [ -f $PROJ_ROOT_PATH/data/models/mobilenetssd_qint8_mixed_float16_1.mm ];then
    echo "mobilenetssd.mm model already exists."
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
