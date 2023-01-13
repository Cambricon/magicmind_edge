#!/bin/bash
set -e
set -x

# 0. convert model
cd $PROJ_ROOT_PATH/export_model 
bash run.sh

# 1. gen_model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode batch_size
bash run.sh qint8_mixed_float16 1

# 2.1 build infer_cpp and infer
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
#bash run.sh quant_mode batch_size
bash run.sh qint8_mixed_float16 1

### 3.eval and perf
#bash $PROJ_ROOT_PATH/benchmark/eval.sh shape_mutable batch_size
bash $PROJ_ROOT_PATH/benchmark/eval.sh qint8_mixed_float16 1
#bash $PROJ_ROOT_PATH/benchmark/perf.sh quant_mode batch_size
bash $PROJ_ROOT_PATH/benchmark/perf.sh qint8_mixed_float16 1

###4. compare eval and perf result
# check 
python ${MAGICMIND_EDGE}/utils/check_result.py
