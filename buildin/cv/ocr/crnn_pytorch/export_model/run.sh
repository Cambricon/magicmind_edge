#!/bin/bash
set -e
set -x
# BATCH_SIZE=$1

if [ -d $PROJ_ROOT_PATH/data/models ];
then
    echo "folder $PROJ_ROOT_PATH/data/models already exist!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/models"
fi

if [ ! -f $PROJ_ROOT_PATH/data/models/crnn_traced.pt ];
then
    # 1.下载数据集
    bash get_datasets.sh
    
    # 2.下载crnn.pytorch实现源码
    cd $PROJ_ROOT_PATH/export_model
    if [ -d "crnn-pytorch" ];
    then
      echo "crnn-pytorch already exists."
    else
      echo "git clone crnn-pytorch..."
      git clone https://github.com/GitYCC/crnn-pytorch.git
      cp crnn-pytorch/checkpoints/crnn_synth90k.pt $PROJ_ROOT_PATH/data/models/
    fi
    
    # 3.trace model
    echo "export model begin..."
    python $PROJ_ROOT_PATH/export_model/export.py --model_weight $PROJ_ROOT_PATH/data/models/crnn_synth90k.pt \
    					      --input_width 100 \
    					      --input_height 32 \
    					      --traced_pt $PROJ_ROOT_PATH/data/models/crnn_traced.pt
    echo "export model end..."
fi
