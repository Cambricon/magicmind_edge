#bin/bash
set -e

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-'1'}

MODEL_NAME=yolov3_tiny_caffe_${QUANT_MODE}_${BATCH_SIZE}.mm

mkdir -p $PROJ_ROOT_PATH/data/models

$UTILS_PATH/rpc_server/start_rpc_server.sh

if [ -f $PROJ_ROOT_PATH/data/models/${MODEL_NAME} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/models/${MODEL_NAME} already exist!"
else 
    echo "generate Magicmind model begin..."
    cd $PROJ_ROOT_PATH/gen_model/
    python gen_model.py \
        --quant_mode ${QUANT_MODE} \
        --batch_size $BATCH_SIZE \
        --caffe_prototxt  $PROJ_ROOT_PATH/data/models/yolov3-tiny.prototxt \
        --caffe_model $PROJ_ROOT_PATH/data/models/yolov3-tiny.caffemodel \
        --datasets_dir $COCO_DATASETS_PATH/val2017 \
        --mm_model $PROJ_ROOT_PATH/data/models/${MODEL_NAME}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/models/${MODEL_NAME}"
fi



        
