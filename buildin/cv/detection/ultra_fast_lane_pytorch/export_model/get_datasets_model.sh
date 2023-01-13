#!/bin/bash
set -e
set -x

# 下载数据集
if [ -z "$TUSIMPLE_DATASETS_PATH" ]; then
    echo "variables TUSIMPLE_DATASETS_PATH not define, please run [source env.sh] first."
    exit 1 
fi
if [ ! -d $TUSIMPLE_DATASETS_PATH ]; then
    mkdir -p $TUSIMPLE_DATASETS_PATH
fi
cd $TUSIMPLE_DATASETS_PATH
if [ ! -d "clips" ];
then
    echo "Downloading test_set.zip"
    wget -c https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip
    unzip -o test_set.zip
else
    echo "val2017 already exists."
fi
if [ ! -f "test_label.json" ];
then
    echo "Downloading test_label.json from https://github.com/TuSimple/tusimple-benchmark/issues/3"
    wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json
else
    echo "test_label.json already exists."
fi

# 下载权重文件
mkdir -p $PROJ_ROOT_PATH/data/models
cd $PROJ_ROOT_PATH/data/models
if [ -f "tusimple_18.pth" ];
then 
    echo "tusimple_18.pth already exists."
else 
    echo "Downloading tusimple_18.pt file"
    gdown -c https://drive.google.com/uc?id=1WCYyur5ZaWczH15ecmeDowrW30xcLrCn
fi

# git clone 
cd $PROJ_ROOT_PATH/export_model
if [ -d "Ultra-Fast-Lane-Detection" ];
then
    echo "Ultra-Fast-Lane-Detection already exists."
    cd Ultra-Fast-Lane-Detection/
else
    echo 'git clone Ultra-Fast-Lane-Detection'
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
fi

# 数据集file_list生成
if [ ! -f $TUSIMPLE_DATASETS_PATH/test.txt ];
then 
    cd $PROJ_ROOT_PATH/export_model
    python convert_tusimple.py --root $TUSIMPLE_DATASETS_PATH
fi

# model patch
if grep -q "PostNet" $PROJ_ROOT_PATH/export_model/Ultra-Fast-Lane-Detection/model/model.py;
then
	echo "PostNet op already exists.";
else
	echo "add PostNet op in 'Ultra-Fast-Lane-Detection/model/model.py'"
    cd $PROJ_ROOT_PATH/export_model
    cp model.patch $PROJ_ROOT_PATH/export_model/Ultra-Fast-Lane-Detection
    cd $PROJ_ROOT_PATH/export_model/Ultra-Fast-Lane-Detection
    git apply model.patch
fi