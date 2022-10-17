#!/bin/bash
set -e

QUANT_MODE=${1:-"qint8_mixed_float16"} #$2 #forced_float32/forced_float16/qint8_mixed_float16
BATCH_SIZE=${2:-1}
PARAMETER_ID=${3:-0}

if [ ! -f $PROJ_ROOT_PATH/data/models/nnUNet_${QUANT_MODE}_${BATCH_SIZE}bs_${PARAMETER_ID}.mm ];then
    # 使用远程3226进行量化，请设置环境变量REMOTE_IP
    echo "generate Magicmind model begin..."
    #export REMOTE_IP=192.168.100.131
    if [ -n "$REMOTE_IP" ]; then
        ../../../utils/rpc_server/start_rpc_server.sh
        python gen_model.py \
            --pt_model $PROJ_ROOT_PATH/data/models/saved_pts/${BATCH_SIZE}bs/2dunet_${PARAMETER_ID}.pt \
            --remote_addres "$REMOTE_IP" \
            --output_model  $PROJ_ROOT_PATH/data/models/nnUNet_${QUANT_MODE}_${BATCH_SIZE}_${PARAMETER_ID}.mm \
            --calib_data_path $PROJ_ROOT_PATH/data/models/saved_pts/${BATCH_SIZE}bs/calib_data_${PARAMETER_ID}.pt \
            --quant_mode ${QUANT_MODE} \
            --batch_size ${BATCH_SIZE} 
        echo "nnUNet_${QUANT_MODE}_${BATCH_SIZE}bs_${PARAMETER_ID}.mm model saved in data/models/"
    else
        echo "$REMOTE_IP is not define"
        exit 1
    fi
else
    echo "mm_model: $PROJ_ROOT_PATH/data/models/nnUNet_${QUANT_MODE}_${BATCH_SIZE}_${PARAMETER_ID}.mm already exist."
fi
    
