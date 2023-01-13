#!/bin/bash
set -e

QUANT_MODE=$1 #force_float32/force_float16/qint8_mixed_float16
BATCH_SIZE=$2
# 使用远程3226进行量化，请设置环境变量REMOTE_IP
echo "generate Magicmind model begin..."
if [ ! -f $MODEL_PATH/pose_body25_${QUANT_MODE}_${BATCH_SIZE}.mm ];
then
    if [ -n "$REMOTE_IP" ]; then
        ${UTILS_PATH}/rpc_server/start_rpc_server.sh
        # BODY_25
        python $PROJ_ROOT_PATH/gen_model/gen_model.py --caffe_prototxt $MODEL_PATH/pose_deploy.prototxt \
                                                    --caffe_model $MODEL_PATH/pose_iter_584000.caffemodel \
                                                    --mm_model $MODEL_PATH/pose_body25_${QUANT_MODE}_${BATCH_SIZE}.mm \
                                                    --batch_size $BATCH_SIZE \
                                                    --quant_mode ${QUANT_MODE} \
                                                    --datasets_dir $COCO_DATASETS_PATH/val2017 \
                                                    --calibrate_list $PROJ_ROOT_PATH/gen_model/calibrate_list.txt \
                                                    --remote_addres "$REMOTE_IP"
    else
        echo "REMOTE_IP not set, please set it first."
        exit 1 
    fi
else
    echo "mm_model: $MODEL_PATH/pose_body25_${QUANT_MODE}_${BATCH_SIZE}.mm already exist."
fi


if [ ! -f $MODEL_PATH/pose_coco_${QUANT_MODE}_${BATCH_SIZE}.mm ];
then
    if [ -n "$REMOTE_IP" ]; then
        ${UTILS_PATH}/rpc_server/start_rpc_server.sh
        # COCO
        python $PROJ_ROOT_PATH/gen_model/gen_model.py --caffe_prototxt $MODEL_PATH/pose_deploy_linevec.prototxt \
                                                    --caffe_model $MODEL_PATH/pose_iter_440000.caffemodel \
                                                    --mm_model $MODEL_PATH/pose_coco_${QUANT_MODE}_${BATCH_SIZE}.mm \
                                                    --batch_size $BATCH_SIZE \
                                                    --quant_mode ${QUANT_MODE} \
                                                    --datasets_dir $COCO_DATASETS_PATH/val2017 \
                                                    --calibrate_list $PROJ_ROOT_PATH/gen_model/calibrate_list.txt \
                                                    --remote_addres "$REMOTE_IP"
    else
        echo "REMOTE_IP not set, please set it first."
        exit 1 
    fi
else
    echo "mm_model: $MODEL_PATH/pose_coco_${QUANT_MODE}_${BATCH_SIZE}.mm already exist."
fi