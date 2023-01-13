#!/bin/bash
set -e

EXPORT(){
  if [ -f $PROJ_ROOT_PATH/data/models/yolov5m_traced.pt ]; then
      echo "yolov5m_traced.pt model already exists."
  else 
      echo "export model begin..."
      BATCH_SIZE=$1
      python $PROJ_ROOT_PATH/export_model/yolov5/export.py --weights $PROJ_ROOT_PATH/data/models/yolov5m.pt --imgsz 640 640 --include torchscript --batch-size ${BATCH_SIZE}
      echo "export model end..."
  fi
}

# 1.下载数据集
cd $PROJ_ROOT_PATH/export_model/
bash get_datasets.sh

# 2.下载权重文件
mkdir -p $PROJ_ROOT_PATH/data/models
cd $PROJ_ROOT_PATH/data/models
if [ -f "yolov5m.pt" ];
then 
    echo "yolov5m.pt already exists."
else 
    echo "Downloading yolov5m.pt file"
    wget -c https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt
fi

# 3.下载yolov5源码,切换到v6.1分支
cd $PROJ_ROOT_PATH/export_model
if [ -d "yolov5" ];
then
    echo "yolov5 already exists."
    cd yolov5
else
    echo 'git clone yolov5'
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    git checkout 3752807c0b8af03d42de478fbcbf338ec4546a6c
fi

# 4.patch-yolov5

if [ -f "yolov5_v6.1_pytorch.patch" ];
then 
    echo "patch file already exit"
else
    cp $PROJ_ROOT_PATH/export_model/yolov5_v6.1_pytorch.patch ./
    git apply yolov5_v6.1_pytorch.patch
fi
cd ..

# 5.patch-torch-cocodataset
if grep -q "SiLU" /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py;
then
	echo "SiLU activation operator already exists.";
else
	echo "add SiLU op in '/usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py and activation.py'"
        patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py < $PROJ_ROOT_PATH/export_model/init.patch
        patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < $PROJ_ROOT_PATH/export_model/activation.patch
fi

# 6.trace model
EXPORT 1

