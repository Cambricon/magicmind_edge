#!/bin/bash
set -e

EXPORT(){
  if [ -f $PROJ_ROOT_PATH/data/models/yolov7_traced.pt ]; then
      echo "yolov7_traced.pt model already exists."
  else 
      echo "export model begin..."
      BATCH_SIZE=$1
      python $PROJ_ROOT_PATH/export_model/yolov7/export.py --weights $PROJ_ROOT_PATH/data/models/yolov7.pt --imgsz 640 640 --include torchscript --batch-size ${BATCH_SIZE}
      echo "export model end..."
  fi
}

# 1.下载数据集
#./get_datasets.sh

# 2.下载权重文件
mkdir -p $PROJ_ROOT_PATH/data/models
cd $PROJ_ROOT_PATH/data/models
if [ -f "yolov7.pt" ];
then 
    echo "yolov7.pt already exists."
else 
    echo "Downloading yolov7.pt file"
    wget -c https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
fi

# 3.下载yolov7源码
cd $PROJ_ROOT_PATH/export_model
if [ -d "yolov7" ];
then
  echo "yolov7 already exists."
else
  echo "git clone yolov7..."
  git clone https://github.com/WongKinYiu/yolov7.git
  cd yolov7
  git checkout 072f76c72c641c7a1ee482e39f604f6f8ef7ee92
fi

# 4.patch-yolov7
if grep -q "ignore_detect_layer = False" $PROJ_ROOT_PATH/export_model/yolov7/models/yolo.py;
then 
  echo "patch already applied!"
else
  echo "modifying the yolov7..."
  cd $PROJ_ROOT_PATH/export_model
  patch -u yolov7/models/yolo.py < yolov7_pytorch.patch
fi

# 5.patch-torch-cocodataset
if grep -q "SiLU" /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py;
then
	echo "SiLU activation operator already exists.";
else
	echo "add SiLU op in '/usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py and activation.py'"
        patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py < $PROJ_ROOT_PATH/export_model/init.patch
        patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < $PROJ_ROOT_PATH/export_model/activation.patch
fi

# 6.export model
cd $PROJ_ROOT_PATH/export_model 
if [ ! -f $MODEL_PATH/yolov7_traced.pt ];
then
    echo "export model begin..."
    cp gen_trancemodel.py ./yolov7/
    cd yolov7
    python gen_trancemodel.py
    echo "export model end..."
fi

