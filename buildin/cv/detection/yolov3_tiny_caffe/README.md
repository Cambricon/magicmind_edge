# yolov3_tiny_caffe

MagicMind是面向寒武纪 MLU 的推理加速引擎。MagicMind能将 Tensorflow, PyTorch, ONNX等训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。本sample探讨如何使用将yolov3网络的Caffe模型转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

## 目录

* [模型概述](#1-模型概述)
* [前提条件](#2-前提条件)
* [快速使用](#3-快速使用)
  * [环境准备](#31-环境准备)
  * [下载仓库](#32-下载仓库)
  * [下载数据集模型](#33-下载数据集模型)
  * [生成MagicMind模型](#34-生成magicmind模型)
  * [编译运行](#35-编译运行)
  * [一键运行](#36-一键运行)
* [高级说明](#4-高级说明)
  * [gen_model参数说明](#41-gen_model参数说明)
  * [infer_cpp代码说明](#42-infer_cpp代码说明)
* [精度和性能benchmark](#5-精度和性能benchmark)
* [免责声明](#6-免责声明)
* [Release Notes](#7-release-notes)

## 1 模型概述

本例使用的yolov3-tiny实现来自github开源项目https://github.com/pjreddie/darknet。下面将展示如何将该项目中Caffe实现的yolov3-tiny模型转换为MagicMind的模型。

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
cd magicmind_edge/buildin/cv/detection/yolov3_tiny_caffe/
```

在开始运行代码前需要先检查env.sh里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集模型

- 下载模型

```bash
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg --no-check-certificate
wget https://pjreddie.com/media/files/yolov3.weights --no-check-certificate

```

- 下载模型
  将darknet原生的yolov3.cfg和yolv3.weight转换为本仓库所需要的yolov3.caffemodel和yolov3.prototxt，请参考[这里](http://gitlab.software.cambricon.com/neuware/software/solutionsdk/caffe_yolo_magicmind)。

### 3.4 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 1
```

结果：

```bash
Generate model done, model save to yolov3_caffe/data/models/yolov3_caffe_model_force_float32_false_1
```

### 3.5 编译运行

编译推理代码:
```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```
推理运行:
```
cd $PROJ_ROOT_PATH/infer_cpp
#params: quant_mode batch_size
bash run.sh qint8_mixed_float16 1
```

### 3.6 一键运行

以上3.3~3.5的步骤也可以通过运行./run.sh来实现一键执行

## 4 高级说明

### 4.1 gen_model参数说明

Caffe yolov3模型转换为MagicMind yolov3模型分成以下几步：

* 使用MagicMind Parser模块将caffe文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:

* `CAFFEMODEL`: yolov3 caffe的权重路径。
* `PROTOTXT`: yolov3 caffe的网络结构路径。
* `MM_MODEL`: 保存MagicMind模型路径。
* `DATASET_DIR`: 校准数据文件路径。
* `QUANT_MODE`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `SHAPE_MUTABLE`: 是否生成可变batch_size的MagicMind模型。
* `BATCH_SIZE`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。
* `DEV_ID`: 设备号。

### 4.2 infer_cpp代码说明

概述：
本例使用MagicMind C++ API编写了名为infer_cpp的视频检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的yolov3目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:

* infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。

参数说明:

* device_id: MLU设备号
* batch_size: 模型batch_size
* magicmind_model: MagicMind模型路径。
* image_dir: 数据集路径
* label_path：coco.names文件
* output_img_dir:推理输出-画框图像路径
* output_pred_dir：推理输出-结果文件路径
* save_imgname_dir：推理输出-所有经过推理的图像名称会被放置于一个名称为image_name.txt文件当中，用于精度验证。
* save_img：是否存储推理输出画框图像 1 存储 0 不存储
* save_pred:是否存储推理结果txt文件 1 存储 0 不存储

## 5 精度和性能benchmark

一键运行benchmark里的脚本：

```bash
# 精度测试
cd $PROJ_ROOT_PATH/benchmark
./eval.sh

# 性能测试
./perf.sh
```

## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* yolov3 caffemodel file下载链接：https://www.dropbox.com/s/bf5z2jw1pg07c9n/yolov3_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0
* yolov3 prototxt file下载链接：https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/yolov3.prototxt
* coco数据集下载链接： http://images.cocodataset.org/zips/val2017.zip

## 7 Release Notes
@TODO
