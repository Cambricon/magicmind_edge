# ModelZoo Edge 

## 1.介绍

MagicMind是面向寒武纪 MLU 的推理加速引擎。

MagicMind能将AI框架(Tensorflow,PyTorch,ONNX,Caffe等)训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本仓库展示如何将CV分类、检测、分割、NLP、语音等场景的前沿和经典模型，通过MagicMind转换和优化，进而运行在基于MagicMind的推理加速引擎的寒武纪加速板卡上的示例程序，为开发者提供丰富的AI应用移植参考。

## 2.前提条件

- Linux 常见操作系统版本(如 Ubuntu16.04，Ubuntu18.04，CentOS7.x 等)，安装 docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪 300 系列及以上的智能加速卡，并安装好驱动(>=v4.20.6)；
- 若不具备以上软硬件条件，可前往[寒武纪开发者社区](https://developer.cambricon.com/)申请试用;

## 3.环境准备

若基于寒武纪云平台环境可跳过该环节。否则需运行以下步骤：

1.请前往[寒武纪开发者社区](https://developer.cambricon.com/)下载 MagicMind(version >= 0.13.0)镜像，名字如下：

magicmind_version_os.tar.gz, 例如 magicmind_0.13.1-1_ubuntu18.04.tar.gz

2.加载：

```bash
docker load -i magicmind_version_os.tar.gz
```

3.运行：

```bash
docker run -it --name=dockername \
           --network=host --cap-add=sys_ptrace \
           -v /your/host/path/MagicMind:/MagicMind \
           -v /usr/bin/cnmon:/usr/bin/cnmon \
           --device=/dev/cambricon_dev0:/dev/cambricon_dev0 --device=/dev/cambricon_ctl \
           -w /MagicMind/ magicmind_version_image_name:tag_name /bin/bash
```

## 4.网络支持列表和链接

### CV：

#### Detection:

| MODELS                                                                     | FRAMEWORK | MLU-3226 | CPP |
| ------------------------------------------------------------------------   | --------- | -------- | --- |
| [C3D](buildin/cv/detection/c3d_caffe)                                      | Caffe     | Yes      | Yes | 
| [Centernet_pytorch](buildin/cv/detection/centernet_pytorch)                | Pytorch   | Yes      | Yes | 
| [Mobilenet-SSD](buildin/cv/detection/mobilenet_ssd_caffe)                  | Caffe     | Yes      | Yes | 
| [Refinedet](buildin/cv/detection/refinedet_caffe)                          | Caffe     | Yes      | Yes | 
| [Retinaface](buildin/cv/detection/retinaface_pytorch)                      | PyTorch   | Yes      | Yes | 
| [Ultra_Fast_Lane_Detection](buildin/cv/detection/ultra_fast_lane_pytorch)  | PyTorch   | Yes      | Yes |
| [YOLOV3](buildin/cv/detection/yolov3_caffe)                                | Caffe     | Yes      | Yes | 
| [YOLOV3 Tiny](buildin/cv/detection/yolov3_tiny_caffe)                      | Caffe     | Yes      | Yes | 
| [YOLOV3_v8](buildin/cv/detection/yolov3_v8_pytorch)                        | PyTorch   | Yes      | Yes | 
| [YOLOV4-mish](buildin/cv/detection/yolov4_mish_caffe)                      | Caffe     | Yes      | Yes | 
| [YOLOV5_v6.1](buildin/cv/detection/yolov5_v6.1_pytorch)                    | PyTorch   | Yes      | Yes |
| [YOLOV7](buildin/cv/detection/yolov7_pytorch)                              | PyTorch   | Yes      | Yes |

#### Classification:

| MODELS                                                                     | FRAMEWORK | MLU-3226 | CPP |
| ------------------------------------------------------------------------   | --------- | -------- | --- |
| [AlexNet](buildin/cv/classification/alexnet_bn_caffe)                      | Caffe     | Yes      | Yes | 
| [Arcface](buildin/cv/classification/arcface_pytorch)                       | PyTorch   | Yes      | Yes | 
| [DenseNet121](buildin/cv/classification/densenet121_caffe)                 | Caffe     | Yes      | Yes | 
| [DenseNet201](buildin/cv/classification/densenet201_caffe)                 | Caffe     | Yes      | Yes | 
| [Googlenet_bn](buildin/cv/classification/googlenet_bn_caffe)               | Caffe     | Yes      | Yes | 
| [Inceptionv2](buildin/cv/classification/inceptionv2_caffe)                 | Caffe     | Yes      | Yes | 
| [Inceptionv3](buildin/cv/classification/inceptionv3_caffe)                 | Caffe     | Yes      | Yes | 
| [Inceptionv4](buildin/cv/classification/inceptionv4_caffe)                 | Caffe     | Yes      | Yes | 
| [Mobilenet-SSD](buildin/cv/detection/mobilenet_ssd_caffe)                  | Caffe     | Yes      | Yes | 
| [Mobilenetv2](buildin/cv/classification/mobilenetv2_caffe)                 | Caffe     | Yes      | Yes | 
| [Mobilenetv3](buildin/cv/classification/mobilenetv3_pytorch)               | Pytorch   | Yes      | Yes | 
| [Resnet50](buildin/cv/classification/resnet50_caffe)                       | Caffe     | Yes      | Yes | 
| [Resnext50](buildin/cv/classification/resnext50_caffe)                     | Caffe     | Yes      | Yes | 
| [Senet50](buildin/cv/classification/senet50_caffe)                         | Caffe     | Yes      | Yes | 
| [Squeezenet1.0](buildin/cv/classification/squeezenet_v1_0_caffe)           | Caffe     | Yes      | Yes | 
| [Squeezenet1.1](buildin/cv/classification/squeezenet_v1_1_caffe)           | Caffe     | Yes      | Yes | 
| [VGG16](buildin/cv/classification/vgg16_caffe)                             | Caffe     | Yes      | Yes | 


#### Segmentation:

| MODELS                                                                     | FRAMEWORK | MLU-3226 | CPP |
| ------------------------------------------------------------------------   | --------- | -------- | --- |
| [DeepLabv3](buildin/cv/segmentation/deeplabv3_tf)                          | Tensorflow| Yes      | Yes | 
| [nnUNet](buildin/cv/segmentation/nnUNet_pytorch)                           | Pytorch   | Yes      | Yes | 
| [SegNet](buildin/cv/segmentation/segnet_caffe)                             | Caffe     | Yes      | Yes | 

#### OCR:

| MODELS                                                                     | FRAMEWORK | MLU-3226 | CPP |
| ------------------------------------------------------------------------   | --------- | -------- | --- |
| [CRNN](buildin/cv/ocr/crnn_pytorch)                                        | Pytorch   | Yes      | Yes | 
| [DBnet](buildin/cv/ocr/dbnet_pytorch)                                      | Pytorch   | Yes      | Yes | 

### Others:

| MODELS                                                                     | FRAMEWORK | MLU-3226 | CPP |
| ------------------------------------------------------------------------   | --------- | -------- | --- |
| [Openpose](buildin/cv/other/opense_caffe)                                  | Caffe     | Yes      | Yes | 


## 5.issues/wiki/forum 跳转链接

## 6.contrib 指引和链接

## 7.LICENSE

ModelZoo Edge 的 License 具体内容请参见[LICENSE](LICENSE)文件。

## 8.免责声明
ModelZoo 仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于 ModelZoo, ModelZoo 也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在 ModelZoo 上，或者您希望更新 ModelZoo 中属于您的数据集或模型，请您通过 Gitee 中提交 issue ，您也可以联系 ecosystem@cambricon.com 告知我们。

# Release notes

## v1.2

### 新增内容

1. 新增 CRNN 网络MagicMind edge支持
2. 新增 DBnet 网络MagicMind edge支持
3. 新增 Retinaface 网络MagicMind edge支持
4. 新增 Refinedet 网络MagicMind edge支持
5. 新增 Openpose 网络MagicMind edge支持
6. 新增 Ultra-Fast-Lane-Detection 网络MagicMind edge支持
7. 新增 Googlenet 网络MagicMind edge支持
8. 新增 YOLOV7 网络MagicMind edge支持

### Bug fix 

1. 修复YOLOV5 由前后处理导致的精度问题
2. 修复所有模型 README.MD 中描述问题以及一键运行脚本相关bug
