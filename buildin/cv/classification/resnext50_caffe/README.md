# resnext50_caffe

MagicMind是面向寒武纪 MLU 的推理加速引擎。MagicMind能将 Tensorflow, PyTorch, ONNX 等训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本sample探讨如何将resnext50网络的Caffe实现转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

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
  * [gen_model细节说明](#41-gen_model细节说明)
  * [infer_cpp细节说明](#42-infer_cpp细节说明)
* [免责声明](#5-免责声明)
* [Release Notes](#6-release-notes)

## 1 模型概述

本例使用的resnext50模型来自https://github.com/soeaver/caffe-model/tree/08979265fc1e1931cef27f40038d492167165804/cls。
下面将展示如何将该项目中Caffe实现的resnext50模型转换为MagicMind的模型。

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
cd magicmind_edge/buildin/cv/classification/resnext50_caffe
```
在开始运行代码前需要先检查env.sh里的环境变量，并且执行以下命令：
```bash
source env.sh
```

### 3.3 下载数据集模型

下载数据集:
```
cd $IMAGENET_DATASETS_PATH
# 请自行从https://image-net.org/challenges/LSVRC/下载LSVRC_2012验证集并解压至$IMAGENET_DATASETS_PATH/ILSVRC2012_img_val目录
```

下载权重:

请用户自行从下述github链接下载模型权重至$PROJ_ROOT_PATH/data/models/.

[deploy_resnext50-32x4d.prototxt & caffemodel](https://github.com/soeaver/caffe-model/tree/master/cls#performance-on-imagenet-validation )

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

### 3.5生成MagicMind模型

```
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_model batch_size
bash run.sh qint8_mixed_float16 1
```

或手动执行一下命令:

```
#!/bin/bash
# 使用远程3226进行量化，请设置环境变量REMOTE_IP
export REMOTE_IP=your remote ip
if [ -n "$REMOTE_IP" ]; then
    ../../../utils/rpc_server/start_rpc_server.sh
    python gen_model.py -r "$REMOTE_IP"
else
# 使用Host侧370量化,则无需设置
    python gen_model.py
fi
```

### 3.5 编译运行

进入infer_cpp目录，在当前目录编译生成可执行文件edge_infer:

```
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```
推理:

```
cd $PROJ_ROOT_PATH/infer_cpp
bash ./bin/edge_infer \
        --magicmind_model $PROJ_ROOT_PATH/data/models/resnext50_${QUANT_MODE}_${BATCH_SIZE}.mm \
        --image_dir $IMAGENET_DATASETS_PATH/ \
        --output_dir $PROJ_ROOT_PATH/data/images
```

参数解析：
* magicmind_model 输入模型路径
* image_dir 测试数据路径
* output_dir 检测结果保存路径

### 3.7 精度验证

```
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

```
top 1 accuracy: 0.771940
top 2 accuracy: 0.869540
top 3 accuracy: 0.904720
top 4 accuracy: 0.923620
top 5 accuracy: 0.934920
```
### 3.8 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/resnext50_qint8_mixed_float16_1.mm --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

### 3.9 一键运行

步骤3.2至3.8可通过脚本./run.sh来实现一键执行.

## 4 高级说明

### 4.1 gen_model细节说明
Caffe resnext50模型转换为MagicMind resnext50模型分成以下几步：
* 使用MagicMind Parser模块将caffe文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `caffe_model`: caffe的权重路径。
* `prototxt`: caffe的网络结构路径。
* `output_model`: 保存MagicMind模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行分类任务。
* `quant_mode`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `batch_size`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。

### 4.2 infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的resnext50图像分类(图像预处理=>推理=>后处理)。
* infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_process.hpp, pre_process.cpp 前处理

参数说明:
* magicmind_model: MagicMind模型路径。
* image_dir: 数据集路径
* output_dir:推理输出-画框图像路径

## 5 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* resnext50 prototxt file下载链接：https://github.com/soeaver/caffe-model/tree/08979265fc1e1931cef27f40038d492167165804/cls
* resnext50 caffemodel 权重下载链接: https://github.com/soeaver/caffe-model/tree/master/cls#performance-on-imagenet-validation

## 6 Release Notes
@TODO
