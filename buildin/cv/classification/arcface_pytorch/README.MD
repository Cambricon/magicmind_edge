# arcface_pytorch

MagicMind 是面向寒武纪 MLU 推理加速引擎。MagicMind 能将 AI 框架(Tensorflow, PyTorch, Caffe, ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何将 Pytorch 人脸识别网络 arcface 转换为 MagicMind模型，进而部署在寒武纪CE3226板卡上。

## 目录

* [模型概述](#1-模型概述)
* [前提条件](#2-前提条件)
* [目录结构](#3-目录结构)
* [快速使用](#4-快速使用)
  * [环境准备](#41-环境准备)
  * [下载仓库](#42-下载仓库)
  * [设置环境变量](#43-设置环境变量)
  * [一键运行](#44-一键运行)
  * [下载数据集和模型](#45-下载数据集和模型)
  * [模型转换](#46-模型转换)
  * [生成MagicMind模型](#47-生成magicmind模型)
  * [编译运行](#48-编译运行)
  * [精度验证](#49-精度验证)
  * [MM_RUN_BENCHMARK性能结果](#410-mm_run_benchmark性能结果)

* [高级说明](#5-高级说明)
  * [gen_model代码解释](#51-gen_model代码解释)
  * [infer_cpp代码解释](#52-infer_cpp代码解释)
* [免责声明](#6-免责声明)

## 1 模型概述

本例使用的arcface模型来自github开源项目 https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

下面将展示如何将该项目中Pytorch框架下arcface模型转换为MagicMind的模型。

## 2 前提条件

请移至[主页面 README.md](README.md)的`2.前提条件`

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
* gen_model: 主要涉及模型量化和转为mm engine过程，要求内部能够一键执行完成该模块完整功能；
* data: 用于暂存测试结果，保存模型等;
* env.sh: 环境变量，需要在模型运行之前source env.sh；
* export_model: 用于生成torch.jit.trace模型;
* infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；
* run.sh: 顶层一键执行脚本，串联各个部分作为整个sample的一键运行脚本；

## 4 快速使用

### 4.1 环境准备

请移至[主页面 README.md](README.md)的`3.环境准备`

### 4.2 下载仓库

```bash
# 下载仓库
git clone http://gitlab.software.cambricon.com/neuware/software/ae/ecosystem/modelzoo/magicmind_edge.git
cd magicmind_edge/buildin/cv/classification/arcface_pytorch
```
### 4.3 设置环境变量

运行前，请检查以下路径，或执行```source env.sh```

```
export PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export MAGICMIND_EDGE="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export IJB_DATASETS_PATH=$MAGICMIND_EDGE/datasets/IJB_release
export UTILS_PATH=$MAGICMIND_EDGE/buildin/cv/utils
export THIRD_PARTY=$MAGICMIND_EDGE/buildin/cv/3rdparty
export MM_RUN_PATH=/mps/bin
```
### 4.4 一键运行


以下4.5~4.10步骤，均可以通过：
```
bash run.sh
``` 
实现一键运行，该脚本包含数据集、模型的自动下载，模型转换与生成，推理代码编译与运行，性能精度测试流程。

**注:** 

一键运行脚本中包含模型与数据集的自动下载，部分数据集与模型由于网络、链接等原因无法自动下载时，需要用户根据README.MD中链接地址手动下载到指定位置。

下面对一键运行脚本中各个步骤进行逐步解析。

### 4.5 下载数据集和模型

- 下载数据集

本例使用[IJB](https://www.nist.gov/itl/iad/ig/ijb-c-dataset-request-form)数据集对模型精度进行验证.

数据集参考下载地址:

https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view

或

https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg

请将IJB数据集下载并解压至```$MAGICMIND_EDGE/datasets/```目录.参考命令如下:

```
cd $MAGICMIND_EDGE/datasets/
tar -xf ijb-*.tar 
mv IJB_* IJB  
```

* 下载模型权重
本例使用MS1MV3训练的backborn为r100的arcface模型进行实验,模型权重下载链接可参考:

https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

或

https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC

```
mkdir -p $PROJ_ROOT_PATH/data/models/
```
请将ms1mv3_arcface_r100_fp16目录中的backbone.pth 下载至```$PROJ_ROOT_PATH/data/models/```目录.


### 4.6 模型转换

```
cd $PROJ_ROOT_PATH/export_model
#param: quant_mode batch_size
bash run.sh qint8_mixed_float16 1
```

### 4.7 编译生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#param: quant_mode batch_size
bash run.sh qint8_mixed_float16 1
```

上述 run.sh 脚本中默认使用远程3226卡板进行模型生成，即设置了**REMOTE_IP**环境变量。

当本地为 MLU370 环境时，可直接执行以下命令进行模型生成:

```bash 
python gen_model.py \
    --pt_model $PROJ_ROOT_PATH/data/models/arcface_r100.pt \
    --output_model_path  $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm \
    --image_dir ./calibrate_list.txt  \
    --quant_mode ${QUANT_MODE} \
    --batch_size ${BATCH_SIZE} 
```

注：
1. 本实例支持batch_size设置功能，已测试最大规模为256。
2. 结果默认保存在$PROJ_ROOT_PATH/data/models文件夹。
3. 有关更多生成MagicMind模型参数相关设置,请参考[寒武纪Magicmind用户手册](https://www.cambricon.com/docs/sdk_1.7.0/magicmind_0.13.1/user_guide/index.html)。

### 4.8 编译运行

进入infer_cpp目录，在当前目录编译生成可执行文件`edge_infer`:

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```

推理：

```bash 
#parameter: quantmodel batchsize
bash run.sh qint8_mixed_float16 
```  

### 4.9 精度验证

```
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：
```bash
IJB-C(1E-5)          IJB—C(1E-4)
{'1e-5': '94.90208', '1e-4': '96.59968'}
```
### 4.10 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/arcface_qint8_mixed_float16_1.mm --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16_1.mm 1
```

## 5 高级说明

### 5.1 gen_model代码解释
Pytorch arcface 模型转换为MagicMind，其流程主要分为以下两步:

* 将原始pth模型通过torch.jit.trace生成固化模型(*.pt).
* 使用MagicMind Parser模块将torch.jit.trace生成的pt文件解析为MagicMind网络结构.
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

gen_model.py参数说明:
* `pt_model`: 转换后pt的路径。
* `image_dir`: 输入图像file_list,保存输入图像的路径。
* `output_model_path`: 保存MagicMind模型路径。
* `quant_mode`: 量化模式，如force_float32，force_float16，qint8_mixed_float16。
* `batch_size`: batch大小，默认为1。
* `remote_addres`: (可选)远端IP地址，是否采用远端量化方式。


### 5.2 infer_cpp代码解释
概述:
本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的arcface人脸识别程序(图像预处理=>推理=>后处理)。相关代码存放在infer_cpp目录下可供参考。其中程序主要由以下内容构成:

* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_precess.hpp, pre_precess.cpp: 前处理。

## 6 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* IJB数据集下载链接:https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view
* IJB数据集下载链接:https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg
* arcface GITHUB下载链接:https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
* arcface backborn模型下载链接:https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC
