#!/bin/bash
if [ -d $PROJ_ROOT_PATH/data/voc_pred ];then
    echo "voc_pred file folder already exists."
else
    mkdir $PROJ_ROOT_PATH/data/voc_pred
fi
rm $PROJ_ROOT_PATH/data/voc_pred/*
python $UTILS_PATH/voc_preds_convert.py 
python $UTILS_PATH/eval_voc.py --path=$PROJ_ROOT_PATH/data/voc_pred/
