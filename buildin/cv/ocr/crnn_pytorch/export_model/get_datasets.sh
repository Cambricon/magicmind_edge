#!/bin/bash
set -e
set -x

if [ ! -d $MJ_DATASETS_PATH ];
then
    mkdir -p $MJ_DATASETS_PATH
fi
cd $MJ_DATASETS_PATH
if [ ! -d "mjsynth_mini" ];
then 
  echo "Downloading mjsynth.zip"
  wget -c --no-check-certificate https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
  tar zxvf mjsynth.tar.gz
else 
  echo "mjsynth already exists."
fi
