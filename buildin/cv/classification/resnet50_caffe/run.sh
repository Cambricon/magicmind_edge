#!/bin/bash
set -e

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi
QUANT_MODE=qint8_mixed_float16 #forced_float32/forced_float16/qint8_mixed_float16
BATCH_SIZE=1

###1.D download caffe model
if [ -e $PROJ_ROOT_PATH/data/models/cnn-models_cvgj.zip ];then
    echo "caffe model already exists."
else
   mkdir -p  $PROJ_ROOT_PATH/data/models
   pushd $PROJ_ROOT_PATH/data/models/
   wget -c https://github.com/cvjena/cnn-models/releases/download/v1.0/cnn-models_cvgj.zip
   unzip -o cnn-models_cvgj.zip
   popd
fi

###2. generate magicmind model#
if [ -f $PROJ_ROOT_PATH/data/models/resnet50_${QUANT_MODE}_${BATCH_SIZE}.mm ];then
    echo "The mm model exit"
else
    cd $PROJ_ROOT_PATH/gen_model
    ## BUILD_MODEL qint8_mixed_float16  batch_size 
    bash run.sh ${QUANT_MODE} ${BATCH_SIZE}
fi

###3. compile the folder: infer_cpp

cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh

###4. infer

if [ -f $PROJ_ROOT_PATH/data/images/* ]; then
    rm $PROJ_ROOT_PATH/data/images/*
fi

cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh ${QUANT_MODE} ${BATCH_SIZE}

###5. compute accuracy top1/top5
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

##6. benchmark test
cd $PROJ_ROOT_PATH/benchmark
## bash perf.sh quant_mode batch_size 
bash perf.sh ${QUANT_MODE} ${BATCH_SIZE} 

###7. check 
python ${MAGICMIND_EDGE}/utils/check_result.py
