#bin/bash
set -e

QUANT_MODE=${1:-'qint8_mixed_float16'}
BATCH_SIZE=${2:-'1'}

MODEL_NAME=yolov3_tiny_caffe_${QUANT_MODE}_${BATCH_SIZE}.mm
if [ -d "$COCO_DATASETS_PATH" ]; then
    echo "COCO_DATASETS already exists."
else
    mkdir -p $COCO_DATASETS_PATH
fi 

cd $COCO_DATASETS_PATH

if [ ! -d "val2017" ];
then
    echo "Downloading val2017.zip"
    wget -c http://images.cocodataset.org/zips/val2017.zip
    unzip -o val2017.zip
else
    echo "val2017 already exists."
fi

if [ ! -d "annotations" ];
then
    echo "Downloading annotations_trainval2017.zip"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -o annotations_trainval2017.zip
else
    echo "annotations_trainval2017 already exists."
fi

if [ ! -d $PROJ_ROOT_PATH/data/models ];
then
    mkdir -p "$PROJ_ROOT_PATH/data/models"
fi

cd $PROJ_ROOT_PATH/data/models
if [ -f "yolov3-tiny.prototxt" -a "yolov3-tiny.caffemodel" ]; 
then
    echo "yolov3-tiny caffemodel/prototxt already exists."
else
    echo "caffemodel/prototxt does not exist."
    echo "please refer to README.MD 4.5 or http://gitlab.software.cambricon.com/neuware/software/solutionsdk/caffe_yolo_magicmind"
    exit 1
fi

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
        --remote_addres "$REMOTE_IP" \
        --mm_model $PROJ_ROOT_PATH/data/models/${MODEL_NAME}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/models/${MODEL_NAME}"
fi



        
