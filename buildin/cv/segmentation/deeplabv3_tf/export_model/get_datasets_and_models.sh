#!/bin/bash
set -e
set -x

if [ -d $PROJ_ROOT_PATH/data/models ];
then
    echo "folder $PROJ_ROOT_PATH/data/models  already exists"
else
    mkdir $PROJ_ROOT_PATH/data/models
fi

cd $PROJ_ROOT_PATH/data/models
if [ -f "frozen_inference_graph.pb" ];
then
  echo "deeplabv3 model already exists."
else
  echo "Downloading deeplabv3 model file"
  wget -c http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz 
  tar -zxvf deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
  mv deeplabv3_mnv2_pascal_train_aug/* ./
  rm -r deeplabv3_mnv2_pascal_train_aug/
fi

cd $MAGICMIND_EDGE/datasets/
if [ ! -d VOCdevkit ];
then
  echo "Downloading VOCtrainval_11-May-2012.tar"
  wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xf VOCtrainval_11-May-2012.tar    
else
  echo "Datasets already exists."
fi

