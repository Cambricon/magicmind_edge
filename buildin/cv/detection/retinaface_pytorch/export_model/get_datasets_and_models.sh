#!/bin/bash
set -e
set -x

if [ ! -d $WIDER_VAL_PATH/images ];
then 
  echo "Downloading WIDER_val.zip"
  cd $MAGICMIND_EDGE/datasets/
  gdown -c https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q -O WIDER_val.zip
  unzip -o WIDER_val.zip
else 
  echo "WIDER_VAL already exists."ls
fi

if [ -d $PROJ_ROOT_PATH/data/models ];
then
    echo "folder $PROJ_ROOT_PATH/data/models already exist!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/models"
fi
cd $PROJ_ROOT_PATH/data/models
if [ -f "Resnet50_Final.pth" ]; 
then
  echo "Resnet50_Final.pth already exists."
else
  echo "Downloading Resnet50_Final.pth file"
  gdown -c https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW -O Resnet50_Final.pth
fi
