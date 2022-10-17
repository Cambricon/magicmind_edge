#!/bin/bash
set -e

QUANT_MODE=$1 #forced_float32/forced_float16/qint8_mixed_float16
BATCH_SIZE=$2

if [ ! -f $PROJ_ROOT_PATH/data/models/vgg_${QUANT_MODE}_${BATCH_SIZE}.mm ];then
    # 使用远程3226进行量化，请设置环境变量REMOTE_IP
    echo "generate Magicmind model begin..."
    #export REMOTE_IP=192.168.100.131
    if [ -n "$REMOTE_IP" ]; then
        ../../../utils/rpc_server/start_rpc_server.sh
        python gen_model.py \
            --prototxt $PROJ_ROOT_PATH/data/models/VGG_ILSVRC_16_layers_deploy.prototxt \
            --caffe_model $PROJ_ROOT_PATH/data/models/VGG_ILSVRC_16_layers.caffemodel \
            --remote_addres "$REMOTE_IP" \
            --output_model_path  $PROJ_ROOT_PATH/data/models/vgg16_${QUANT_MODE}_${BATCH_SIZE}.mm \
            --image_dir $IMAGENET_DATASETS_PATH/ \
            --quant_mode ${QUANT_MODE} \
     	    --batch_size ${BATCH_SIZE} 
    else
        echo "REMOTE_IP not set, please set it first."
        exit 1 
    fi
fi    
