#!/bin/bash
python ${UTILS_PATH}/imagenet_eval.py --ground_truth $PROJ_ROOT_PATH/data/val_2015.txt --output $PROJ_ROOT_PATH/data/images/
