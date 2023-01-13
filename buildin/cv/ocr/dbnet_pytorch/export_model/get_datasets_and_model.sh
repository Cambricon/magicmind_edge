#!/bin/bash
set -e
set -x

if [ ! -d $TEXT_DATASETS_PATH ];
then
    echo "Downloading total_text"
    cd $MAGICMIND_EDGE/datasets/
    echo "Please download datasets from https://drive.google.com/drive/folders/12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7"
    exit 1
    unzip -o total_text.zip
else 
    echo "total_text already exists."
fi

if [ ! -d $TEXT_DATASETS_PATH/test_images ];
then
    cd $TEXT_DATASETS_PATH
        echo "Please download test_images from https://drive.google.com/uc?id=1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2&export=download"
        exit 1
        unzip - o totaltext.zip
        mv Images/Test ./test_images
        rm -rf totaltext.zip __MACOSX Images
else 
    echo "test_images already exists."
fi

if [ -d $PROJ_ROOT_PATH/data/models ];
then
    echo "folder $PROJ_ROOT_PATH/data/models already exist!!!"
else
    mkdir -p "$PROJ_ROOT_PATH/data/models"
fi

cd $PROJ_ROOT_PATH/data/models
if [ -f "totaltext_resnet18" ]; 
then
    echo "totaltext_resnet18 already exists."
else
    gdown https://drive.google.com/uc?id=1bVBYXSnFpwZxu8wHROAqJ4ddhLfZE7MY
fi
