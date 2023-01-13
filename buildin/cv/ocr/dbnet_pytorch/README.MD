# Dbnet_pytorch

MagicMind是面向寒武纪 MLU 推理加速引擎。MagicMind 能将 AI 框架(Tensorflow, PyTorch, Caffe, ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 Dbnet 网络的 PyTorch 实现转换为 MagicMind 模型，进而部署在寒武纪CE3226板卡上。

## 目录

- [Dbnet\_pytorch](#dbnet_pytorch)
  - [目录](#目录)
  - [1 模型概述](#1-模型概述)
  - [2 前提条件](#2-前提条件)
  - [3 目录结构](#3-目录结构)
  - [4 快速使用](#4-快速使用)
    - [4.1 环境准备](#41-环境准备)
    - [4.2 下载仓库](#42-下载仓库)
    - [4.3 设置环境变量](#43-设置环境变量)
    - [4.4 一键运行](#44-一键运行)
    - [4.5 下载数据集](#45-下载数据集)
    - [4.6 模型转换](#46-模型转换)
    - [4.7 生成MagicMind模型](#47-生成magicmind模型)
    - [4.8 编译运行](#48-编译运行)
    - [4.9 精度验证](#49-精度验证)
    - [4.10 MM\_RUN\_BENCHMARK性能结果](#410-mm_run_benchmark性能结果)
  - [5 高级说明](#5-高级说明)
    - [5.1 export\_model细节说明](#51-export_model细节说明)
    - [5.2 gen\_model参数说明](#52-gen_model参数说明)
    - [5.3 infer\_cpp代码说明](#53-infer_cpp代码说明)
  - [6 免责声明](#6-免责声明)

## 1 模型概述
本例使用的Dbnet实现来自github开源项目[https://github.com/MhLiao/DB](https://github.com/MhLiao/DB)。

下面将展示如何将该项目中PyTorch实现的Dbnet网络转换为MagicMind的模型。

## 2 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3 目录结构

```
.
|-- README.MD
|-- benchmark
|-- data
|-- env.sh
|-- gen_model
|-- infer_cpp
`-- run.sh
```
目录结构说明：
* benchmark: 提供mm_run测试脚本，用于测试该模型在不同输入规模、不同数据精度、不同硬件设备下的性能；同时提供精度验证功能；
* data: 用于暂存测试结果，保存模型等；
* gen_model: 主要涉及模型量化和转为mm engine过程，要求内部能够一键执行完成该模块完整功能；
* env.sh: 环境变量，需要在模型运行之前source env.sh；
* infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；
* run.sh: 顶层一键执行脚本，串联各个部分作为整个sample的一键运行脚本；

## 4 快速使用
### 4.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 4.2 下载仓库

* 下载仓库
```bash
git clone http://gitlab.software.cambricon.com/neuware/software/ae/ecosystem/modelzoo/magicmind_edge.git 
cd magicmind_edge/buildin/cv/ocr/dbnet_pytorch/
```
### 4.3 设置环境变量

运行前，请检查以下路径，或执行```source env.sh```：

```bash
export PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export MAGICMIND_EDGE="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export TEXT_DATASETS_PATH=$MAGICMIND_EDGE/datasets/total_text
export UTILS_PATH=$MAGICMIND_EDGE/buildin/cv/utils
export THIRD_PARTY=$MAGICMIND_EDGE/buildin/cv/3rdparty
export MM_RUN_PATH=/mps/bin/
```

注意：数据集默认保存在 **$MAGICMIND_EDGE/datasets/** 路径下，**请用户自行创建或挂载**。

设置**REMOTE_IP**与**REMOTE_DIR**

```bash
export REMOTE_IP=*   #edge端ip地址
export REMOTE_DIR=*  #magicdmind 挂载在edge端的路径
```
### 4.4 一键运行

以下4.5~4.10的步骤，均可以通过：
```
bash run.sh
``` 
实现一键运行，该脚本包含数据集、模型的自动下载，模型转换与生成，推理代码编译与运行，性能精度测试流程。

**注:** 

一键运行脚本中包含模型与数据集的自动下载，部分数据集与模型由于网络、链接等原因无法自动下载时，需要用户根据README.MD中链接地址手动下载到指定位置。

下面对一键运行脚本中各个步骤进行逐步解析。

### 4.5 下载数据集
```bash
cd $PROJ_ROOT_PATH/export_model 
./get_datasets_and_model.sh
```

### 4.6 模型转换
```bash
cd $PROJ_ROOT_PATH/export_model 
./run.sh
```

### 4.7 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#parameter: quant_model batch_size
bash run.sh qint8_mixed_float16 1
```

上述 run.sh 脚本中默认使用远程3226卡板进行模型生成，即设置了**REMOTE_IP**环境变量。

当本地为 MLU370 环境时，可直接执行以下命令进行模型生成:
```
python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $PROJ_ROOT_PATH/data/models/dbnet_traced.pt \
                                                      --output_model $PROJ_ROOT_PATH/data/models/dbnet_${QUANT_MODE}_${BATCH_SIZE}.mm \
                                                      --image_dir $PROJ_ROOT_PATH/data/data_test \
                                                      --quant_mode ${QUANT_MODE} \
                                                      --batch_size ${BATCH_SIZE}
```

生成的Magicmind模型默认保存在在 $PROJ_ROOT_PATH/data/models 文件夹中。

有关更多生成MagicMind模型参数相关设置,请参考[寒武纪Magicmind用户手册](https://www.cambricon.com/docs/sdk_1.7.0/magicmind_0.13.1/user_guide/index.html)。

### 4.8 编译运行

进入infer_cpp目录，在当前目录编译生成可执行文件`./bin/edge_infer`:

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```
推理：
```bash 
#parameter: quant_model batchsize
bash run.sh qint8_mixed_float16 1
```

### 4.9 精度验证

```bash
cd $PROJ_ROOT_PATH/benchmark
#parameter: quant_model batchsize
bash eval.sh
```
结果：
```bash
[INFO] [2022-12-01 07:53:39,792] precision : 0.881504 (300)
[INFO] [2022-12-01 07:53:39,792] recall : 0.773159 (300)
[INFO] [2022-12-01 07:53:39,792] fmeasure : 0.823784 (1)
```
### 4.10 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/dbnet_qint8_mixed_float16_1.mm --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

## 5 高级说明
### 5.1 export_model细节说明
向MagicMind导入PyTorch模型首先需要先使用torch.jit.trace生成PyTorch模型文件。 

```bash
cd $PROJ_ROOT_PATH/export_model
# 使用$PROJ_ROOT_PATH/export_model/DB/demo.py jit.trace导出模型。

cd $PROJ_ROOT_PATH/export_model/DB
    python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml \
                  --image_path $TEXT_DATASETS_PATH/test_images/img10.jpg \
                  --resume $PROJ_ROOT_PATH/data/models/totaltext_resnet18 \
                  --polygon --box_thresh 0.7 --visualize \
                  --traced_pt $PROJ_ROOT_PATH/data/models/dbnet_traced.pt
```

### 5.2 gen_model参数说明
PyTorch dbnet网络转换为MagicMind模型可分为以下几个步骤:
1.通过torch.jit.trace生成pt文件。
2.使用MagicMind Parser模块将torch.jit.trace生成的pt文件解析为MagicMind网络结构。
3.模型量化(可选)。
4.使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `pt_model`: 转换后pt的路径。
* `output_model`: 保存MagicMind模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行文字识别任务。
* `quant_mode`: 量化模式，如force_float32，force_float16，qint8_mixed_float16。
* `shape_mutable`: 是否生成可变batch_size的MagicMind模型。
* `batch_size`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。

### 5.3 infer_cpp代码说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的文字识别程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的dbnet文字识别(图像预处理=>推理=>后处理)。相关代码存放在infer_cpp目录下可供参考。其中程序主要由以下内容构成:
* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_precess.hpp, pre_precess.cpp: 前处理。
* post_precess.hpp, post_precess.cpp: 后处理。

参数说明:
* `magicmind_model`: MagicMind模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行文字识别任务。
* `batch_size`: 推理时，输入tensor的维度

## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* total_text 数据集链接：https://github.com/cs-chan/Total-Text-Dataset
* Dbnet实现源码和权重下载链接：https://github.com/MhLiao/DB
