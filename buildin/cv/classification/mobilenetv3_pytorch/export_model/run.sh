#bin/bash
set -e
if [ ! -f $PROJ_ROOT_PATH/data/models/mobilenet-v3_small.torchscript.pt ];then
  python convert.py $PROJ_ROOT_PATH/data/models/mobilenetv3_small_67.4.pth.tar
fi 

