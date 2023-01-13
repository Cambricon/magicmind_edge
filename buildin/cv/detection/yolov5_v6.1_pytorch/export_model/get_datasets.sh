#!/bin/bash
set -e
set -x

if [ -z "$COCO_DATASETS_PATH" ]; then
    echo "variables COCO_DATASETS_PATH not define, please run [source env.sh] first."
    exit 1 
fi
if [ ! -d $COCO_DATASETS_PATH ]; then
# 下载数据集
    mkdir -p $COCO_DATASETS_PATH
fi
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
