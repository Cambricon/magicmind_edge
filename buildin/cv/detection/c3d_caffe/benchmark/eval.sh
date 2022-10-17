#!/bin/bash

python $UTILS_PATH/compute_top1_and_top5.py \
    --result_label_file $PROJ_ROOT_PATH/data/output/eval_labels.txt \
    --result_1_file $PROJ_ROOT_PATH/data/output/eval_result_1.txt \
    --result_5_file $PROJ_ROOT_PATH/data/output/eval_result_5.txt \
    --top1andtop5_file $PROJ_ROOT_PATH/data/output/eval_result.txt
