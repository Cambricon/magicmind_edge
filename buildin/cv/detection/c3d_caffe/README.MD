# c3d_caffe

MagicMind 是面向寒武纪 MLU 推理加速引擎。MagicMind 能将 AI 框架(Tensorflow, PyTorch, Caffe, ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 c3d 网络的 Caffe 实现转换为 MagicMind 模型，进而部署在寒武纪CE3226板卡上。

## 目录
- [c3d\_caffe](#c3d_caffe)
  - [目录](#目录)
  - [1 模型概述](#1-模型概述)
  - [2 前提条件](#2-前提条件)
  - [3 目录结构](#3-目录结构)
  - [4 快速使用](#4-快速使用)
    - [4.1 环境准备](#41-环境准备)
    - [4.2 下载仓库](#42-下载仓库)
    - [4.3 设置环境变量](#43-设置环境变量)
    - [4.4 一键运行](#44-一键运行)
    - [4.5 下载数据集和模型](#45-下载数据集和模型)
    - [4.6 生成MagicMind模型](#46-生成magicmind模型)
    - [4.7 编译运行](#47-编译运行)
    - [4.8 精度验证](#48-精度验证)
    - [4.9 MM\_RUN\_BENCHMARK性能结果](#49-mm_run_benchmark性能结果)
  - [5 高级说明](#5-高级说明)
    - [5.1 gen\_model细节说明](#51-gen_model细节说明)
    - [5.2 infer\_cpp细节说明](#52-infer_cpp细节说明)
  - [6 免责声明](#6-免责声明)

* [高级说明](#5-高级说明)
  * [gen_model细节说明](#51-gen_model细节说明)
  * [infer_cpp细节说明](#52-infer_cpp细节说明)
* [免责声明](#6-免责声明)

## 1 模型概述

本例使用的c3d实现来自github开源项目https://github.com/facebookarchive/C3D/tree/master/C3D-v1.1
 
下面将展示如何将该项目中Caffe实现的c3d模型转换为MagicMind的模型。

## 2 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3 目录结构

```
.
|-- README.MD
|-- benchmark
|-- data
|-- env.sh
|-- export_model
|-- gen_model
|-- infer_cpp
`-- run.sh
```
目录结构说明：
* benchmark: 提供mm_run测试脚本，用于测试该模型在不同输入规模、不同数据精度、不同硬件设备下的性能；同时提供精度验证功能；
* data: 用于暂存测试结果，保存模型等;
* gen_model: 主要涉及模型量化和转为mm engine过程，要求内部能够一键执行完成该模块完整功能；
* env.sh: 环境变量，需要在模型运行之前source env.sh；
* export_model: 用于下载模式和数据集；
* infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；
* run.sh: 顶层一键执行脚本，串联各个部分作为整个sample的一键运行脚本；

## 4 快速使用

### 4.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 4.2 下载仓库
```bash
# 下载仓库
git clone http://gitlab.software.cambricon.com/neuware/software/ae/ecosystem/modelzoo/magicmind_edge.git
cd magicmind_edge/buildin/cv/detection/c3d_caffe
```
### 4.3 设置环境变量

运行前，请检查以下路径，或执行```source env.sh```：

```
export MAGICMIND_EDGE="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export THIRD_PARTY=$MAGICMIND_EDGE/buildin/cv/3rdparty
export UFC101_DATASETS_PATH=$MAGICMIND_EDGE/datasets/UCF-101
export UTILS_PATH=$MAGICMIND_EDGE/buildin/cv/utils
```

注意：数据集默认保存在 **$MAGICMIND_EDGE/datasets/** 路径下，**请用户自行创建或挂载**。

设置**REMOTE_IP**与**REMOTE_DIR**
```
export REMOTE_IP=*   #edge端ip地址
export REMOTE_DIR=*  #magicdmind 挂载在edge端的路径
```

### 4.4 一键运行

以下4.5~4.9的步骤，均可以通过：
```
bash run.sh
``` 
实现一键运行，该脚本包含数据集、模型的自动下载，模型转换与生成，推理代码编译与运行，性能精度测试流程。

**注:** 

一键运行脚本中包含模型与数据集的自动下载，部分数据集与模型由于网络、链接等原因无法自动下载时，需要用户根据README.MD中链接地址手动下载到指定位置。

下面对一键运行脚本中各个步骤进行逐步解析。

### 4.5 下载数据集和模型

- 下载数据集

```
cd $PROJ_ROOT_PATH/export_model
bash get_datasets_and_models.sh
``` 

### 4.6 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#parameter: quant_mode batch_size
bash run.sh qint8_mixed_float16 1
```
上述 run.sh 脚本中默认使用远程3226卡板进行模型生成，即设置了**REMOTE_IP**环境变量。

当本地为 MLU370 环境时，可直接执行以下命令进行模型生成:
```
python gen_model.py \
    --quant_mode ${QUANT_MODE} \
    --batch_size $BATCH_SIZE \
    --caffe_prototxt $PROJ_ROOT_PATH/data/models/c3d_resnet18_r2_ucf101.prototxt \
    --caffe_model $PROJ_ROOT_PATH/data/models/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel \
    --datasets_dir ${UFC101_DATASETS_PATH} \
    --mm_model $PROJ_ROOT_PATH/data/models/${MODEL_NAME}
```

生成的Magicmind模型默认保存在在$PROJ_ROOT_PATH/data/models文件夹中。

有关更多生成MagicMind模型参数相关设置,请参考[寒武纪Magicmind用户手册](https://www.cambricon.com/docs/sdk_1.7.0/magicmind_0.13.1/user_guide/index.html)。

### 4.7 编译运行

进入infer_cpp目录，在当前目录编译生成可执行文件`./bin/edge_infer`:

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```
推理：
```bash 
#parameter: quantmodel batchsize
bash run.sh qint8_mixed_float16 1
```  

### 4.8 精度验证

```bash
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：
```bash
top1 accuracy: 0.836664
top5 accuracy: 0.998847
```
### 4.9 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/c3d_qint8_mixed_float16_1.mm --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

## 5 高级说明
### 5.1 gen_model细节说明
Caffe c3d模型转换为MagicMind c3d模型分成以下几步：
* 使用MagicMind Parser模块将caffe文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `caffe_model`: c3d caffe的权重路径。
* `prototxt`: c3d caffe的网络结构路径。
* `output_model_path`: 保存MagicMind模型路径。
* `image_dir`: 校准数据文件路径。
* `quant_model`: 量化模式，如force_float32，force_float16，qint8_mixed_float16。
* `batch_size`: batch大小，默认为1。
* `remote_addres`: (可选)远端IP地址，是否采用远端量化方式。

### 5.2 infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的视频检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的c3d视频捕捉(视频帧采样=>推理=>后处理)。其中程序主要由以下内容构成:
* infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。

参数说明:
* resized_w: 预处理相关参数。指定图像预处理中缩放大小。
* resized_h: 预处理相关参数。指定图像预处理中缩放大小。
* magicmind_model: MagicMind模型路径。
* video_list: 输入视频列表文件，文件中每一行为一个视频文件路径。
* output_dir: 动作识别结果保存目录。每一个视频片段的top5识别结果将保存为一个txt文件。
* sampling_rate: 视频帧采样频率，默认为2，意味着每两帧采样一帧。
* clip_step: 截取视频片段的移动步长，默认为-1，若为-1，clip_step等于CLIP_LEN * sampling_rate。
例如视频帧按1，2，3，4，5，6编号，clip_step指定为2，则第一个视频片段从编号为1的视频帧开始，第二个视频片段从编号为3的视频帧开始。

## 6 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* c3d caffemodel file下载链接：https://www.dropbox.com/s/bf5z2jw1pg07c9n/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0
* prototxt 下载链接: https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/c3d_resnet18_r2_ucf101.prototxt
* UCF101数据集下载链接： https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar
* UCF101数据集标签下载连接：https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip

