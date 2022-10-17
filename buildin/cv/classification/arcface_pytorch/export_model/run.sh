#!/bin/bash
set -e

# 1.下载数据集
if [ -d $IJB_DATASETS_PATH  ];
then 
    echo "IJB datasets already exists."
else
    echo "Please follow the README.md to download the IJB-B,IJB-C datasets."
fi


# 2.下载权重文件
if [ -f $PROJ_ROOT_PATH/data/models/backborn.pth ];
then 
    echo "The arcface backborn already exists."
else
    echo "Please follow the README.md to download the Model Zoo in ./data/models/"
fi

# 3.下载仓库
if [ -d $PROJ_ROOT_PATH/export_model/insightface ]; then 
    echo "deepinsight already exists."
else
    echo "git clone deepinsight..."
    git clone https://github.com/deepinsight/insightface.git
fi

# 4.trace model
# param: batchsize
if [ -f $PROJ_ROOT_PATH/data/models/arcface_r100.pt ];then
    echo "arcface_r100.pt aleady exists."
else
    python $PROJ_ROOT_PATH/export_model/export.py --weights $PROJ_ROOT_PATH/data/models/backbone.pth --network r100 --output_pt $PROJ_ROOT_PATH/data/models/arcface_r100.pt
fi

