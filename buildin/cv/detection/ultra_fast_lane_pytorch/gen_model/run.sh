#!/bin/bash

QUANT_MODE=$1 #force_float32 #/force_float16/qint8_mixed_float16
BATCH_SIZE=$2
# 使用远程3226进行量化，请设置环境变量REMOTE_IP
echo "generate Magicmind model begin..."
if [ ! -f $PROJ_ROOT_PATH/data/models/tusimple_${QUANT_MODE}_${BATCH_SIZE}.mm ];then
    if [ -n "$REMOTE_IP" ]; then
        ${UTILS_PATH}/rpc_server/start_rpc_server.sh
        python gen_model.py \
            --remote_addres "$REMOTE_IP" \
            --output_model_path  $PROJ_ROOT_PATH/data/models/tusimple_${QUANT_MODE}_${BATCH_SIZE}.mm \
            --image_dir $TUSIMPLE_DATASETS_PATH/clips/0530/1492626047222176976_0 \
            --quant_mode ${QUANT_MODE} \
            --pt_model $PROJ_ROOT_PATH/data/models/tusimple_18_traced.pt \
            --batch_size ${BATCH_SIZE} 
        echo "tusimple_${QUANT_MODE}_${BATCH_SIZE}.mm model saved in data/models/"
    else 
        echo "REMOTE_IP not set, please set it first."
        exit 1 
    fi
else
    echo "mm_model: $PROJ_ROOT_PATH/data/models/tusimple_${QUANT_MODE}_${BATCH_SIZE} already exist."
fi

