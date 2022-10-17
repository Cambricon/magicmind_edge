#!/bin/bash
set -e

QUANT_MODE=$1 #forced_float32/forced_float16/qint8_mixed_float16
BATCH_SIZE=$2
CONF_THRES=$3
IOU_THRES=$4

if [ -f ${PROJ_ROOT_PATH}/data/models/yolov3_${QUANT_MODE}_${BATCH_SIZE}.mm ];then
    echo "yolov3_${QUANT_MODE}_${BATCH_SIZE}.mm model already exists."
else
    # 使用远程3226进行量化，请设置环境变量REMOTE_IP
    echo "generate Magicmind model begin..."
    if [ -n "$REMOTE_IP" ]; then
        echo $(pwd)
        ${UTILS_PATH}/rpc_server/start_rpc_server.sh
        python gen_model.py \
            --remote_addres "$REMOTE_IP" \
            --output_model_path  $PROJ_ROOT_PATH/data/models/yolov3_${QUANT_MODE}_${BATCH_SIZE}.mm \
            --image_dir $COCO_DATASETS_PATH/val2017 \
            --quant_mode ${QUANT_MODE} \
    	    --batch_size ${BATCH_SIZE} \
    	    --conf_thres ${CONF_THRES} \
    	    --iou_thres ${IOU_THRES}
        echo "yolov3_${QUANT_MODE}_${BATCH_SIZE}.mm model saved in data/models/"
    else
        echo "REMOTE_IP not set, please set it first."
        exit 1 
    fi
fi    
