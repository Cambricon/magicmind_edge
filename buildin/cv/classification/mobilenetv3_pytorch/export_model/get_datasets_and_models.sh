#bin/bash
set -e
set -x

ORIGIN_PT='mobilenetv3_small_67.4.pth.tar'
if [ ! -f $PROJ_ROOT_PATH/data/models/$ORIGIN_PT ];then
  echo "downloading mobilenetv3_small_67.4.pth.tar"
  gdown -c https://drive.google.com/uc?id=1lCsN3kWXAu8C30bQrD2JTZ7S2v4yt23C -O $PROJ_ROOT_PATH/data/models/$ORIGIN_PT
fi

if [ ! -f "$PROK_ROOT_PATH/export_model/pytorch-mobilenet-v3.zip" ];then
  echo "pytorch-mobilenet-v3.zip"
  cd $PROJ_ROOT_PATH/export_model/
  wget -c https://github.com/kuan-wang/pytorch-mobilenet-v3/archive/refs/heads/master.zip -O pytorch-mobilenet-v3.zip
  unzip -o pytorch-mobilenet-v3.zip -d $PROJ_ROOT_PATH/export_model
fi

cd $IMAGENET_DATASETS_PATH
echo "Downloading LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/ to $DATASETS_PATH "
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi

