#!/bin/bash
QUANT_MODE=$1  
BATCH_SIZE=$2

python $UTILS_PATH/mj_eval.py --result_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${BATCH_SIZE}/result_file.txt