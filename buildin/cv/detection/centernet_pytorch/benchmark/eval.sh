#!/bin/bash
set -e
set -x
QUANT_MODE=$1
BATCH=$2
IMAGE_NUM=$3
if [ -d $PROJ_ROOT_PATH/data/json ];
then 
    echo "folder $PROJ_ROOT_PATH/data/json already exist."
else
    mkdir $PROJ_ROOT_PATH/data/json
fi 
COMPUTE_COCO(){
    QUANT_MODE=$1
    BATCH=$2
    IMG_NUM=$3
    python $UTILS_PATH/compute_coco_mAP.py --file_list $PROJ_ROOT_PATH/data/file_list.txt \
                                    --result_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${BATCH} \
                                    --ann_dir $COCO_DATASETS_PATH \
                                    --data_type val2017 \
                                    --json_name $PROJ_ROOT_PATH/data/json/centernet_${QUANT_MODE}_${BATCH} \
                                    --image_num ${IMG_NUM}
}

# #dynamic
if [ $# != 0 ];
then 
    COMPUTE_COCO ${QUANT_MODE} ${BATCH} ${IMAGE_NUM}
else  
    echo "Parm doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh 1
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode 1 0.001 0.65 1000
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_cpp
            bash run.sh $quant_mode 1 $batch 1000
            COMPUTE_COCO $quant_mode $batch 1000
            python $MAGICMIND_EDGE/test/compare_eval.py --metric cocomAP --output_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_${quant_mode}_${batch}/log_eval --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_cpp_output_${batch}_log_eval --model centernet_pytorch
        done
    done
fi
