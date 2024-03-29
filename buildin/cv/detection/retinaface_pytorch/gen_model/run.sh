#!/bin/bash

QUANT_MODE=$1
BATCH_SIZE=$2

if [ ! -f $PROJ_ROOT_PATH/data/models/retinaface_${QUANT_MODE}_${BATCH_SIZE}.mm ];then
    # 使用远程3226进行量化，请设置环境变量REMOTE_IP
    echo "generate Magicmind model begin..."
    #export REMOTE_IP=192.168.100.131
    if [ -n "$REMOTE_IP" ]; then
        ${UTILS_PATH}/rpc_server/start_rpc_server.sh
        python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $PROJ_ROOT_PATH/data/models/retinaface_traced.pt \
                                                      --output_model $PROJ_ROOT_PATH/data/models/retinaface_${QUANT_MODE}_${BATCH_SIZE}.mm \
                                                      --image_dir $WIDER_VAL_PATH/images/ \
                                                      --quant_mode ${QUANT_MODE} \
                                                      --batch_size ${BATCH_SIZE} \
                                                      --remote_addres "$REMOTE_IP" 
        echo "retinaface_${QUANT_MODE}_${BATCH_SIZE}.mm model saved in data/models/"
   else
        echo "REMOTE_IP not set, please set it first."
        exit 1 
    fi
else 
     echo "mm_model: $PROJ_ROOT_PATH/data/models/centernet_${QUANT_MODE}_${BATCH_SIZE}.mm already exist."
fi    
