#!/bin/bash
set -e

cd $PROJ_ROOT_PATH
pip install -r requirements.txt
quant_mode=qint8_mixed_float16

###0. download datasets and model
cd $PROJ_ROOT_PATH/export_model
bash get_datasets_model.sh

###1. torch.jit.trace model generate
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###2. build_model#
cd $PROJ_ROOT_PATH/gen_model
# BUILD_MODEL quant_mode batch_size 
bash run.sh $quant_mode 1 

###2. compile the folder: infer_cpp
 cd $PROJ_ROOT_PATH/infer_cpp
 bash build.sh 

###3. infer
cd $PROJ_ROOT_PATH/infer_cpp
# bash run.sh quant_mode batch_size  
bash run.sh $quant_mode 1

###4. compute accuracy 
bash $PROJ_ROOT_PATH/benchmark/eval.sh 

###5. benchmark test
cd $PROJ_ROOT_PATH/benchmark
# bash perf.sh quant_mode batch_size
bash perf.sh $quant_mode 1 

###6. check 
python ${MAGICMIND_EDGE}/utils/check_result.py
