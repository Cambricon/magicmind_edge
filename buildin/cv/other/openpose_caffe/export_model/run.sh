#!/bin/bash
set -e

if [ -z "$COCO_DATASETS_PATH" ]; then
    echo "variables COCO_DATASETS_PATH not define, please run [source env.sh] first."
    exit 1 
fi

# 下载数据集
mkdir -p $COCO_DATASETS_PATH
cd $COCO_DATASETS_PATH

if [ ! -d "val2017" ];
then 
  echo "Downloading val2017.zip"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -o val2017.zip
else 
  echo "val2017 already exists."
fi

if [ ! -d "annotations" ];
then
  echo "Downloading annotations_trainval2017.zip"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -o annotations_trainval2017.zip
else 
  echo "annotations_trainval2017 already exists."
fi

# 2.下载权重文件
mkdir -p $PROJ_ROOT_PATH/data/models
cd $PROJ_ROOT_PATH/data/models
if [ -f "pose_deploy.prototxt" ]; 
then
  echo "pose_deploy.prototxt already exists."
else
  echo "Downloading pose_deploy.prototxt file"
  wget -c https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt
fi

if [ -f "pose_iter_584000.caffemodel" ]; 
then
  echo "pose_iter_584000.caffemodel already exists."
else
  echo "Downloading pose_iter_584000.caffemodel file"
  wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel
fi

if [ -f "pose_deploy_linevec.prototxt" ]; 
then
  echo "pose_deploy_linevec.prototxt already exists."
else
  echo "Downloading pose_deploy_linevec.prototxt file"
  wget -c https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
fi


if [ -f "pose_iter_440000.caffemodel" ]; 
then
  echo "pose_iter_440000.caffemodel already exists."
else
  echo "Downloading pose_iter_440000.caffemodel file"
  wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
fi

