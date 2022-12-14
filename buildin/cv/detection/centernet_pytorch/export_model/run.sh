#!/bin/bash
set -e
set -x
BATCH_SIZE=$1

if [ ! -f $PROJ_ROOT_PATH/data/models/ctdet_coco_dlav0_1x_traced_${BATCH_SIZE}bs.pt ];
then
    # 1.下载数据集和模型
    bash get_datasets_and_models.sh
    
    # 2.下载centernet实现源码
    cd $PROJ_ROOT_PATH/export_model
    if [ -d "CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6" ];
    then
      echo "centernet already exists."
    else
      echo "git clone centernet..."
      wget -c https://github.com/xingyizhou/CenterNet/archive/2b7692c377c6686fb35e473dac2de6105eed62c6.zip
      unzip 2b7692c377c6686fb35e473dac2de6105eed62c6.zip
    fi
    
    # 3.patch-centernet
    if grep -q "if head == 'hm':" $PROJ_ROOT_PATH/export_model/CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/networks/dlav0.py;
    then 
      echo "modifying file: CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/networks/dlav0.py has been already done"
    else
      echo "modifying file: CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/networks/dlav0.py ..."
      cd $PROJ_ROOT_PATH/export_model
      patch CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/networks/dlav0.py < centernet_dlav0.diff
    fi
    
    if grep -q "# from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn" CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/model.py;
    then
      echo "modifying file: CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/model.py has been already done"
    else
      echo "modifying file: CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/model.py ..."
      cd $PROJ_ROOT_PATH/export_model
      patch CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/model.py < centernet_model.diff
    fi
    
    # 5.trace model
    echo "export model begin..."
    python $PROJ_ROOT_PATH/export_model/export.py --model_weight $PROJ_ROOT_PATH/data/models/ctdet_coco_dlav0_1x.pth \
    					      --input_width 512 \
    					      --input_height 512 \
    					      --batch_size ${BATCH_SIZE} \
    					      --traced_pt $PROJ_ROOT_PATH/data/models/ctdet_coco_dlav0_1x_traced_${BATCH_SIZE}bs.pt
    echo "export model end..."
fi
