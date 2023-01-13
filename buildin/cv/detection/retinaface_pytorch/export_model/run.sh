#!/bin/bash
set -e
set -x

if [ ! -f $PROJ_ROOT_PATH/data/models/retinaface_traced.pt ];
then
    # 1.下载数据集和模型
    bash get_datasets_and_models.sh
    
    # 2.下载retinaface实现源码
    cd $PROJ_ROOT_PATH/export_model
    if [ -d "Pytorch_Retinaface-b984b4b775b2c4dced95c1eadd195a5c7d32a60b" ];
    then
      echo "retinafacet already exists."
    else
      echo "git clone retinaface..."
      wget -c https://github.com/biubug6/Pytorch_Retinaface/archive/b984b4b775b2c4dced95c1eadd195a5c7d32a60b.zip
      unzip -o b984b4b775b2c4dced95c1eadd195a5c7d32a60b.zip
    fi
    
    # 3.patch-retinaface

    if grep -q "leaky = 0.0" Pytorch_Retinaface-b984b4b775b2c4dced95c1eadd195a5c7d32a60b/models/net.py;
    then
      echo "modifying file: Pytorch_Retinaface-b984b4b775b2c4dced95c1eadd195a5c7d32a60b/models/net.py has been already done"
    else
      echo "modifying file: Pytorch_Retinaface-b984b4b775b2c4dced95c1eadd195a5c7d32a60b/models/net.py ..."
      cd $PROJ_ROOT_PATH/export_model
      patch -u Pytorch_Retinaface-b984b4b775b2c4dced95c1eadd195a5c7d32a60b/models/net.py < retinaface.diff
    fi

    # 5.trace model
    echo "export model begin..."
    python $PROJ_ROOT_PATH/export_model/export.py --model_weight $PROJ_ROOT_PATH/data/models/Resnet50_Final.pth \
    					      --input_width 672 \
    					      --input_height 1024 \
    					      --traced_pt $PROJ_ROOT_PATH/data/models/retinaface_traced.pt
    echo "export model end..."
fi
