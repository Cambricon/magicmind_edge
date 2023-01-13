#!/bin/bash
set -e

precision=qint8_mixed_float16
batch_size=8

### 1.download datasets 
if [ ! -d $IMAGENET_DATASETS_PATH ];
then
    echo "Please execute : mkdir $MAGICMIND_EDGE/datasets/imagenet"
    echo "Please download imagenet(ILSVRC2012_val) from https://image-net.org/challenges/LSVRC/"
    echo "Images path like : $MAGICMIND_EDGE/datasets/imagenet/ILSVRC2012_val_0000*.JPEG"
    exit 1
else 
    echo "IMAGENET already exists."
fi

if [ ! -d "$PROJ_ROOT_PATH/data/models" ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
if [ ! -d "$PROJ_ROOT_PATH/data/images" ];then
    mkdir -p $PROJ_ROOT_PATH/data/images
fi

### 2.download caffe model
if [ ! -d $PROJ_ROOT_PATH/data/models ];then
    mkdir -p $PROK_ROOT_PATH/data/models
fi
cd $PROJ_ROOT_PATH/data/models
if [ ! -f "googlenet_bn.caffemodel" ];then
  echo "googlenet_bn.caffemodel"
  wget -c  https://github.com/lim0606/caffe-googlenet-bn/blob/master/snapshots/googlenet_bn_stepsize_6400_iter_1200000.caffemodel?raw=true --no-check-certificate -O googlenet_bn.caffemodel
fi 
if [ ! -f "googlenet_bn_deploy.prototxt" ];then
  echo "googlenet_bn_deploy.prototxt"
  wget -c https://raw.githubusercontent.com/lim0606/caffe-googlenet-bn/master/deploy.prototxt --no-check-certificate -O googlenet_bn_deploy.prototxt
fi

echo "modify googlenet_bn_deploy.prototxt"
sed -i "s/layers/layer/" googlenet_bn_deploy.prototxt

### 3.build magicmind model
if [ -f $PROJ_ROOT_PATH/data/models/googlenet_bn_${precision}_${batch_size}.mm ];then
    echo "The mm model exit"
else
    cd $PROJ_ROOT_PATH/gen_model
    ## BUILD_MODEL qint8_mixed_float16 batch_size 
    bash run.sh ${precision} ${batch_size} 
fi

### 4. compile the folder: infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh

### 5 infer_cpp
if [ -f $PROJ_ROOT_PATH/data/images/* ]; then
    cd $PROJ_ROOT_PATH/data/images/
    rm -rf *
fi
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh $precision ${batch_size}

###6. compute accuracy top1/top5
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh

###6. benchmark test
cd $PROJ_ROOT_PATH/benchmark
## bash perf.sh quant_mode batch_size
bash perf.sh $precision ${batch_size}

# check 
python ${MAGICMIND_EDGE}/utils/check_result.py

