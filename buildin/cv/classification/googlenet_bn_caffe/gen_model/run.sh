#!/bin/bash
QUANT_MODE=$1       
BATCH_SIZE=$2

if [ -f $PROJ_ROOT_PATH/data/models/googlenet_bn_${QUANT_MODE}_${BATCH_SIZE}.mm ];then
    echo "googlenet_bn_${QUANT_MODE}_${BATCH_SIZE}.mm already exists."
else
    # 使用远程3226进行量化，请设置环境变量REMOTE_IP
    echo "generate Magicmind model begin..."
    if [ -n "$REMOTE_IP" ]; then
        ../../../utils/rpc_server/start_rpc_server.sh
        python gen_model.py \
            --prototxt $PROJ_ROOT_PATH/data/models/googlenet_bn_deploy.prototxt \
            --caffe_model $PROJ_ROOT_PATH/data/models/googlenet_bn.caffemodel \
            --remote_addres "$REMOTE_IP" \
            --output_model_path $PROJ_ROOT_PATH/data/models/googlenet_bn_${QUANT_MODE}_${BATCH_SIZE}.mm\
            --image_dir $IMAGENET_DATASETS_PATH/ \
            --quant_mode ${QUANT_MODE} \
            --input_width 224 \
            --input_height 224 \
     	    --batch_size ${BATCH_SIZE} 
        echo "googlenet_bn_${QUANT_MODE}_${BATCH_SIZE}.mm model saved in data/models/"
    else 
        echo "REMOTE_IP not set, please set it first."
        exit 1 
    fi
fi    
