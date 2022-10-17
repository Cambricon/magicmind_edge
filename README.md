# ModelZoo Edge 

## 介绍

MagicMind是面向寒武纪MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。

MagicMind能将深度学习框架(Tensorflow,PyTorch,ONNX,Caffe等) 训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本仓库展示如何将CV分类、检测、分割、NLP、语音等场景的前沿和经典模型，通过MagicMind转换和优化，进而运行在基于MagicMind的推理加速引擎的寒武纪加速板卡上的示例程序，为开发者提供丰富的AI应用移植参考。

## 网络支持列表和链接
CV：
| MODELS  | FRAMEWORK | MLU-3226 | CPP |
| ------------- | ------------- | ------------- | ------------- | 
| [AlexNet_with_bn_caffe](buildin/cv/classification/alexnet_bn_caffe) | Caffe | Yes | Yes |
| [Arcface](buildin/cv/classification/arcface_pytorch) | PyTorch | Yes | Yes |
| [C3D](buildin/cv/detection/c3d_caffe/) | Caffe | Yes | Yes |
| [Centernet_pytorch](buildin/cv/detection/centernet_pytorch/) | Pytorch | Yes | Yes |
| [DeepLabv3](buildin/cv/segmentation/deeplabv3_tf/) | Tensorflow | Yes | Yes |
| [DenseNet121](buildin/cv/classification/densenet121_caffe/) | Caffe | Yes | Yes |
| [DenseNet201](buildin/cv/classification/densenet201_caffe/) | Caffe | Yes | Yes |
| [Inceptionv2](buildin/cv/classification/inceptionv2_caffe/) | Caffe | Yes | Yes |
| [Inceptionv3](buildin/cv/classification/inceptionv3_caffe/) | Caffe | Yes | Yes |
| [Inceptionv4](buildin/cv/classification/inceptionv4_caffe/) | Caffe | Yes | Yes |
| [Mobilenet-SSD](buildin/cv/detection/mobilenet_ssd_caffe/) | Caffe | Yes | Yes | 
| [Mobilenetv2](buildin/cv/classification/mobilenetv2_caffe/) | Caffe | Yes | Yes |
| [Mobilenetv3](buildin/cv/classification/mobilenetv3_pytorch/) | Pyotch | Yes | Yes |
| [nnUNet](buildin/cv/segmentation/nnUNet_pytorch/) | Pytorch | Yes | Yes |
| [Resnet50](buildin/cv/classification/resnet50_caffe/) | Caffe | Yes | Yes |
| [Resnext50](buildin/cv/classification/resnext50_caffe/) | Caffe | Yes | Yes |
| [SegNet](buildin/cv/segmentation/segnet_caffe/) | Caffe | Yes | Yes |
| [Senet50](buildin/cv/classification/senet50_caffe/) | Caffe | Yes | Yes |
| [Squeezenet1.0](buildin/cv/classification/squeezenet_v1_0_caffe/) | Caffe | Yes | Yes |
| [Squeezenet1.1](buildin/cv/classification/squeezenet_v1_1_caffe/) | Caffe | Yes | Yes |
| [VGG16](buildin/cv/classification/vgg16_caffe/) | Caffe | Yes | Yes |
| [YOLOV5_v6.1](buildin/cv/detection/yolov5_v6.1_pytorch/) | PyTorch | Yes | Yes |
| [YOLOV3_v8](buildin/cv/detection/yolov3_v8_pytorch/) | PyTorch | Yes | Yes |
| [YOLOV3](buildin/cv/detection/yolov3_caffe/) | Caffe | Yes | Yes |
| [YOLOV4-mish](buildin/cv/detection/yolov4_mish_caffe/) | Caffe | Yes | Yes |

NLP:

| MODELS  | FRAMEWORK | MLU-3226 | CPP |
| ------------- | ------------- | ------------- | ------------- |


## issues/wiki/forum跳转链接

## contrib指引和链接

## LICENSE
ModelZoo Edge的License具体内容请参见[LICENSE](https://gitee.com/cambricon/magicmind_edge/blob/master/LICENSE)文件。

## 免责声明
ModelZoo仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于ModelZoo, ModelZoo也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在ModelZoo上，或者您希望更新ModelZoo中属于您的数据集或模型，请您通过Github或者Gitee中提交issue，您也可以联系ecosystem@cambricon.com告知我们。
