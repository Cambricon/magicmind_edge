#!/bin/bash

python ${UTILS_PATH}/nnUNet_evalute.py --data_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
       --model_path $PROJ_ROOT_PATH/data/models/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1 \
       --ref_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/labelsTr \
       --softmax_output_folder $PROJ_ROOT_PATH/data/softmax_output 
