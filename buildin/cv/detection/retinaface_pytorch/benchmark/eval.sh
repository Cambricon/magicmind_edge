#!/bin/bash
python $UTILS_PATH/widerface_evaluate/evaluation.py \
      -p $PROJ_ROOT_PATH/data/images/pred_txts/ \
      -g $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface-b984b4b775b2c4dced95c1eadd195a5c7d32a60b/widerface_evaluate/ground_truth/
