#!/bin/bash

python $PROJ_ROOT_PATH/data/gen_predlist.py
cd $PROJ_ROOT_PATH/export_model/DB
python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume totaltext_resnet18 --polygon --box_thresh 0.7
python $PROJ_ROOT_PATH/benchmark/prec_log.py
