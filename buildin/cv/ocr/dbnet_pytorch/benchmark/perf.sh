#!/bin/bash
set -e

source ${MAGICMIND_EDGE}/utils/remote_tools.sh

#!/bin/bash
set -e

quant_mode=${1:-'qint8_mixed_float16'}
batchs=${2:-1 4 8}

for batch in ${batchs[@]}; do
    # export model
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    # gen_model
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh qint8_mixed_float16 $batch
    # perf test
    REMOTE_MM_RUN \
        --magicmind_model data/models/dbnet_${quant_mode}_${batch}.mm \
        --input_dims ${batch},736,896,3 \
        --batch ${batch} 
done





