#!/bin/bash
set -e

# 1.下载数据集
bash $PROJ_ROOT_PATH/export_model/get_datasets.sh

# 2.下载权重文件
if [ ! -d $PROJ_ROOT_PATH/data/models ];then
    mkdir "$PROJ_ROOT_PATH/data/models"
fi

cd $PROJ_ROOT_PATH/data/models

if [ -f "yolov3.pt" ];then 
    echo "yolov3.pt already exists."
else 
    echo "Downloading yolov3.pt file"
    wget -c https://github.com/ultralytics/yolov3/releases/download/v8/yolov3.pt
fi

# 3.下载yolov3源码,v8分支
cd $PROJ_ROOT_PATH/export_model
if [ -d "yolov3" ];then
    echo "yolov3 already exists."
    cd yolov3
else
    echo 'git clone yolov3'
    git clone https://github.com/ultralytics/yolov3.git
    cd yolov3
    git checkout v8
fi

# 4.trace model
cd $PROJ_ROOT_PATH/export_model/yolov3
cp $PROJ_ROOT_PATH/export_model/export.py ./
if [ -f $PROJ_ROOT_PATH/data/models/yolov3_traced.pt ];then
    echo "yolov3_traced.pt model already exists."
else
    python $PROJ_ROOT_PATH/export_model/yolov3/export.py --weights $PROJ_ROOT_PATH/data/models/yolov3.pt --imgsz 416 416 --batch_size 1
    echo "torch.jit.trace model for yolov3 saved in $PROJ_ROOT_PATH/data/models/"
fi
