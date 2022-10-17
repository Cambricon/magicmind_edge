#!/bin/bash
set -e
set -x

# get model
cd $PROJ_ROOT_PATH/data/models

if [ -f "Task002_Heart.zip" ];
then
  echo "Task002_Heart.zip already exists."
else
  echo "Downloading models ..."
  wget -c https://zenodo.org/record/4003545/files/Task002_Heart.zip?download=1 -O Task002_Heart.zip
  unzip Task002_Heart.zip
fi

# download datasets
if [ ! -d $NNUNET_DATASETS_PATH ];then
    mkdir -p $NNUNET_DATASETS_PATH
fi

cd $NNUNET_DATASETS_PATH
if [ -d "Task02_Heart" ];
then
  echo "Task02_Heart already exist."
else
  echo "Please follow the README.md to download Task02_Heart.tar to $NNUNET_DATASETS_PATH and tar -xf Task02_Heart.tar."  
fi

