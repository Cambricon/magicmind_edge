#!/bin/bash
set -e
set -x

if [ ! -f $PROJ_ROOT_PATH/data/models/dbnet_traced.pt ];
then
    # 1.下载数据集
    bash get_datasets_and_model.sh
    
    # 2.下载dbnet实现源码
    cd $PROJ_ROOT_PATH/export_model
    if [ -d "DB" ];
    then
      echo "DB already exists."
    else
      echo "git clone DB..."
      git clone https://github.com/MhLiao/DB.git
      pip install -r requirement.txt
      echo "modified clone DB..."
      cp -r DB_diff/cpu_dcn DB/assets/ops/
      cp DB_diff/vision_deform_conv.py DB/backbones/
      cd DB/assets/ops/cpu_dcn
      python setup.py build develop
    fi

    # 3.patch-retinaface
    cd $PROJ_ROOT_PATH/export_model
    if grep -q "traced_model = torch.jit.trace(model, torch.rand(1, 3, 736, 896))" DB/demo.py
    then
      echo "modifying file:demo.py has been already done"
    else
      echo "modifying file: demo.py ..."
      patch -u DB/demo.py < DB_diff/demo.diff
      patch -u DB/experiments/seg_detector/base_totaltext.yaml < DB_diff/totaltext.diff
      patch -u DB/structure/model.py < DB_diff/model.diff
      patch -u DB/backbones/resnet.py < DB_diff/resnet.diff
      patch -u DB/eval.py < DB_diff/eval.diff
      patch -u DB/experiments/seg_detector/totaltext_resnet18_deform_thre.yaml < DB_diff/resnet18_deform_thre.diff
    fi
    # 4.trace model
    echo "export model begin..."
    cd $PROJ_ROOT_PATH/export_model/DB
    python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml \
                  --image_path $TEXT_DATASETS_PATH/test_images/img10.jpg \
                  --resume $PROJ_ROOT_PATH/data/models/totaltext_resnet18 \
                  --polygon --box_thresh 0.7 --visualize \
                  --traced_pt $PROJ_ROOT_PATH/data/models/dbnet_traced.pt
fi
