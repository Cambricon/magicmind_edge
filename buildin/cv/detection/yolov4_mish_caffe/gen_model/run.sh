#bin/bash
set -e

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-'1'}

MODEL_NAME=yolov4_mish_${QUANT_MODE}_${BATCH_SIZE}.mm

if [ -f $PROJ_ROOT_PATH/data/models/${MODEL_NAME} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/models/${MODEL_NAME} already exist!"
else 
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/models/${MODEL_NAME}"
    cd $PROJ_ROOT_PATH/gen_model/
    if [ -n "$REMOTE_IP" ]; then
        ../../../utils/rpc_server/start_rpc_server.sh
        python gen_model.py \
            --remote_addres "$REMOTE_IP" \
            --quant_mode ${QUANT_MODE} \
            --batch_size $BATCH_SIZE \
            --caffe_prototxt  $PROJ_ROOT_PATH/data/models/yolov4-mish.prototxt \
            --caffe_model $PROJ_ROOT_PATH/data/models/yolov4-mish.caffemodel   \
            --datasets_dir $COCO_DATASETS_PATH/val2017 \
            --mm_model $PROJ_ROOT_PATH/data/models/${MODEL_NAME}
        echo "Generate model done, model save to $PROJ_ROOT_PATH/data/models/${MODEL_NAME}"
    else
        echo "REMOTE_IP not set, please set it first."
        exit 1 
    fi
fi
