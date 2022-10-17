#!/bin/bash
set -e

#QUANT_MODE=qint8_mixed_float16
QUANT_MODE=$1 #forced_float32/forced_float16/qint8_mixed_float16
BATCH_SIZE=$2
#BATCH_SIZE=128

if [ ! -f $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm ];then
    # 使用远程3226进行量化，请设置环境变量REMOTE_IP
    echo "generate Magicmind model begin..."
    #export REMOTE_IP=192.168.100.131
    if [ -n "$REMOTE_IP" ]; then
        ../../../utils/rpc_server/start_rpc_server.sh
        python gen_model.py \
            --pt_model $PROJ_ROOT_PATH/data/models/arcface_r100.pt \
            --remote_addres "$REMOTE_IP" \
            --output_model_path  $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm \
            --image_dir ./calibrate_list.txt  \
            --quant_mode ${QUANT_MODE} \
            --batch_size ${BATCH_SIZE} 
        echo "arcface.mm model saved in data/models/"
    else
        echo "$REMOTE_IP is not define"
        exit 1
    fi
else
    echo "mm_model: $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm already exist."
fi
    
