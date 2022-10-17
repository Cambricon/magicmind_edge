# yolov3_caffe

MagicMind是面向寒武纪 MLU 的推理加速引擎。MagicMind能将Tensorflow,PyTorch,ONNX 等训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。本sample探讨如何使用将yolov3网络的Caffe模型转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

## 目录

* [模型概述](#1-模型概述)
* [前提条件](#2-前提条件)
* [快速使用](#3-快速使用)
  * [环境准备](#31-环境准备)
  * [下载仓库](#32-下载仓库)
  * [下载数据集模型](#33-下载数据集模型)
  * [目录结构](#34-目录结构)
  * [生成MagicMind模型](#35-生成magicmind模型)
  * [编译运行](#36-编译运行)
  * [精度验证](#37-精度验证)
  * [MM_RUN_BENCHMARK性能结果](#38-mm_run_benchmark性能结果)
  * [一键运行](#39-一键运行)
* [高级说明](#4-高级说明)
  * [gen_model代码解释](#41-gen_model参数说明)
  * [infer_cpp代码解释](#42-infer_cpp代码说明)
* [免责声明](#5-免责声明)
* [Release Notes](#6-release-notes)

## 1 模型概述

 本例使用的yolov3实现来自github开源项目https://github.com/pjreddie/darknet。下面将展示如何将该项目中Caffe实现的yolov3模型转换为MagicMind的模型。

## 2 前提条件

* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本 MLU370 S4或 MLU370 X4，并安装好驱动(>=v4.20.6)；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 3 快速使用

### 3.1 环境准备

下载MagicMind(version >= 0.13.0)镜像(下载链接待开放)，名字如下：
magicmind_version_os.tar.gz

加载镜像：
```bash
docker load -i xxx.tar.gz
```

运行镜像：

```
docker run -it --name=dockername --network=host --cap-add=sys_ptrace -v /your/host/path/MagicMind:/MagicMind -v /usr/bin/cnmon:/usr/bin/cnmon --device=/dev/cambricon_dev0:/dev/cambricon_dev0 --device=/dev/cambricon_ctl -w /MagicMind/ <image name> /bin/bash
```
### 3.2 下载仓库

```bash
# 下载仓库
git clone git@gitee.com:cambricon/magicmind_edge.git
cd magicmind_edge/buildin/cv/detection/yolov3_caffe/
```

在开始运行代码前需要先检查env.sh里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集模型

- 下载模型

```bash
cd $PROJ_ROOT_PATH/data/models/
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg --no-check-certificate
wget https://pjreddie.com/media/files/yolov3.weights --no-check-certificate

```

- darknet2caffe

将darknet原生的yolov3.cfg和yolv3.weight转换为本仓库所需要的yolov3.caffemodel和yolov3.prototxt，请参考[这里](http://gitlab.software.cambricon.com/neuware/software/solutionsdk/caffe_yolo_magicmind)。

### 3.4 目录结构

```
.
|-- README.md
|-- benchmark
|-- data
|-- env.sh
|-- gen_model
|-- infer_cpp
`-- run.sh
```
目录结构说明：
* benchmark: 提供mm_run测试脚本，用于测试该模型在不同输入规模、不同数据精度、不同硬件设备下的性能；同时提供精度验证功能；

* gen_model: 主要涉及模型量化和转为mm engine过程，要求内部能够一键执行完成该模块完整功能；

* data: 用于暂存测试结果，保存模型等;

* infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；一般图像类及前后处理不复杂的网络，建议要有c++的推理；

* run.sh: 顶层一键执行脚本，串联各个部分作为整个sample的一键运行脚本；


### 3.5 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#param: quant_mode batch_size
bash run.sh qint8_mixed_float16 1
```
### 3.6 编译运行

进入infer_cpp目录，在当前目录编译生成可执行文件`edge_infer`:
```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```

推理：

```bash 
bash ./bin/edge_infer \ #或 bin/host_infer在host端进行推理验证
      --magicmind_model $PROJ_ROOT_PATH/data/models/yolov3_qint8_mixed_float16_1.mm \
      --image_dir $IMAGENET_DATASETS_PATH/ \
      --output_dir $PROJ_ROOT_PATH/data/images 
#参数解析：
      --magicmind_model 输入模型路径
      --image_dir 测试数据路径
      --output_dir 检测结果保存路径
```  

### 3.7 精度验证

```bash
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.678
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.624
```
### 3.8 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/yolov3_qint8_mixed_float16_1.mm --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

### 3.9 一键运行

以上3.2~3.8的步骤，均可以通过```bash run.sh``` 实现一键运行。

**注:** 
* 1. 执行一键运行脚本,需用户确保已下载好模型和数据集; 
* 2. 非云平台用户在本地测试时,可将$MAGICMIND_EDGE路径下**datasets_test**文件软连接为**datasets**,即```ln -s datasets_test datasets```;
* 3. **datasets_test**文件为数据集路径,仅包含少量测试用例/图片,使用于用户进行功能验证,无法确保精度结果; 
* 4. 非云平台用户若需测试完整精度结果,需将本地完整的数据集路径软连接到$MAGICMIND_EDGE/datasets路径下;


## 4 高级说明

### 4.1 gen_model参数说明

Caffe yolov3模型转换为MagicMind yolov3模型分成以下几步：

* 使用MagicMind Parser模块将caffe文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:

* `caffe_model`: yolov3 caffe的权重路径。
* `prototxt`: yolov3 caffe的网络结构路径。
* `output_model_path`: 保存MagicMind模型路径。
* `image_dir`: 校准数据文件路径。
* `quant_model`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `batch_size`: batch大小，默认为1。
* `remote_addres`: (可选)远端IP地址，是否采用远端量化方式。

### 4.2 infer_cpp代码说明

概述：
本例使用MagicMind C++ API编写了名为infer_cpp的视频检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的yolov3目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:
* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_process.hpp, pre_process.cpp: 前处理。


## 5 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* yolov3 caffemodel file下载链接：https://www.dropbox.com/s/bf5z2jw1pg07c9n/yolov3_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0
* yolov3 prototxt file下载链接：https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/yolov3.prototxt
* coco数据集下载链接： http://images.cocodataset.org/zips/val2017.zip

## 6 Release Notes

@TODO
