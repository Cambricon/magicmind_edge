#!/bin/bash
set -e

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

quant_mode=${1:-'qint8_mixed_float16'}
batchs=${2:-1 4 8}

for batch in ${batchs[@]}; do
    # export model
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    # gen_model
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $quant_mode $batch
    # perf test
    cd $PROJ_ROOT_PATH/benchmark
    REMOTE_MM_RUN \
        --magicmind_model data/models/arcface_${quant_mode}_${batch}.mm \
        --batch ${batch} 
done
