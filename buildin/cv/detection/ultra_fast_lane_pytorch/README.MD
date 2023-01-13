# Ultra-Fast-Lane-Detection

MagicMind 是面向寒武纪 MLU 推理加速引擎。MagicMind 能将 AI 框架(Tensorflow, PyTorch, Caffe, ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 Ultra-Fast-Lane-Detecion 从 PyTorch 转换为 MagicMind 模型，进而部署在寒武纪CE3226板卡上。

## 目录

- [Ultra-Fast-Lane-Detection](#ultra-fast-lane-detection)
  - [目录](#目录)
  - [1 模型概述](#1-模型概述)
  - [2 前提条件](#2-前提条件)
  - [3 目录结构](#3-目录结构)
  - [4 快速使用](#4-快速使用)
    - [4.1 环境准备](#41-环境准备)
    - [3.2 下载仓库](#32-下载仓库)
    - [4.3 设置环境变量](#43-设置环境变量)
    - [4.4 一键运行](#44-一键运行)
    - [4.5 下载数据集和模型](#45-下载数据集和模型)
    - [4.6 模型转换](#46-模型转换)
    - [4.7 生成MagicMind模型](#47-生成magicmind模型)
    - [4.8 编译运行](#48-编译运行)
    - [4.9 精度验证](#49-精度验证)
    - [4.10 MM\_RUN\_BENCHMARK性能结果](#410-mm_run_benchmark性能结果)
  - [5 高级说明](#5-高级说明)
    - [5.1 gen\_model参数说明](#51-gen_model参数说明)
    - [5.2 infer\_cpp代码说明](#52-infer_cpp代码说明)
  - [6 免责声明](#6-免责声明)

## 1 模型概述

 本例使用的 Ultra-Fast-Lane-Detecion 实现来自github开源项目https://github.com/cfzd/Ultra-Fast-Lane-Detection.git 。下面将展示如何将该项目中PyTorch实现的模型转换为MagicMind的模型。

## 2 前提条件

请移至[主页面 README.md](README.md)的`2.前提条件`

## 3 目录结构

```
.
|-- README.MD
|-- benchmark
|-- env.sh
|-- export_model
|-- gen_model
|-- infer_cpp
|-- requirements.txt
`-- run.sh
```
目录结构说明：
* benchmark: 提供mm_run测试脚本，用于测试该模型在不同输入规模、不同数据精度、不同硬件设备下的性能；同时提供精度验证功能；
* data: 用于暂存测试结果，保存模型等;
* gen_model: 主要涉及模型量化和转为mm engine过程，要求内部能够一键执行完成该模块完整功能；
* env.sh: 环境变量，需要在模型运行之前source env.sh；
* export_model:用户生成保存torch.jit.trace模型；
* infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；
* requirements.txt: 依赖包；
* run.sh: 顶层一键执行脚本，串联各个部分作为整个sample的一键运行脚本；

## 4 快速使用

### 4.1 环境准备

请移至[主页面 README.md](README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone http://gitlab.software.cambricon.com/neuware/software/ae/ecosystem/modelzoo/magicmind_edge.git
cd magicmind_edge/buildin/cv/detection/ultra_fast_lane_pytorch
```
### 4.3 设置环境变量

开始运行代码前需要先执行以下命令安装必要的库：
```bash
pip install -r requirements.txt
```

运行前，请检查以下路径，或执行```source env.sh```：

```bash
export PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export MAGICMIND_EDGE="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export TUSIMPLE_DATASETS_PATH=$MAGICMIND_EDGE/datasets/Tusimple
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

### 4.5 下载数据集和模型
```bash
# 下载数据集
cd $TUSIMPLE_DATASETS_PATH
wget -c https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip
unzip -o test_set.zip
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json

# 下载权重文件
cd $PROJ_ROOT_PATH/data/models
gdown -c https://drive.google.com/uc?id=1WCYyur5ZaWczH15ecmeDowrW30xcLrCn

# 下载Ultra-Fast-Lane-Detection源码
cd $PROJ_ROOT_PATH/export_model
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git

# 数据集 file_list生成
python convert_tusimple.py --root $TUSIMPLE_DATASETS_PATH
```

### 4.6 模型转换

1. 打入patch 

```bash
cp model.patch $PROJ_ROOT_PATH/export_model/Ultra-Fast-Lane-Detection
cd $PROJ_ROOT_PATH/export_model/Ultra-Fast-Lane-Detection
git apply model.patch
```

注： model.patch 是将后处理加入模型推理中，提升整体性能。模型直接输出的结果是坐标值，此坐标值是图片中的横坐标x，纵坐标为固定的点y，（x，y）为图像上的点，x>0, 该点有效。固定坐标值如下：

```
# tmp_dict['lanes']为横坐标         
tmp_dict['lanes'] = generate_tusimple_lines(out[i],imgs[0,0].shape,griding_num)
# tmp_dict['h_samples']为固定纵坐标
tmp_dict['h_samples'] = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
  270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
  430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
  590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
```

2. jit.trace导出模型，模型默认保存在$PROJ_ROOT_PATH/data/models文件夹下
```bash
cd $PROJ_ROOT_PATH/export_model/
python export_trace.py tusimple.py --test_model $PROJ_ROOT_PATH/data/models/tusimple_18.pth
```

### 4.7 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode batch_size 
bash run.sh qint8_mixed_float16 1 
```
上述 run.sh 脚本中默认使用远程3226卡板进行模型生成，即设置了**REMOTE_IP**环境变量。

当本地为 MLU370 环境时，可直接执行以下命令进行模型生成:

```
python gen_model.py \
    --output_model_path  $PROJ_ROOT_PATH/data/models/tusimple_${QUANT_MODE}_${BATCH_SIZE}.mm \
    --image_dir $TUSIMPLE_DATASETS_PATH/clips/0530/1492626047222176976_0 \
    --quant_mode ${QUANT_MODE} \
    --pt_model $PROJ_ROOT_PATH/data/models/tusimple_18_traced.pt \
    --batch_size ${BATCH_SIZE} 
```

生成的Magicmind模型默认保存在在$PROJ_ROOT_PATH/data/models文件夹中。

有关更多生成MagicMind模型参数相关设置,请参考[寒武纪Magicmind用户手册](https://www.cambricon.com/docs/sdk_1.7.0/magicmind_0.13.1/user_guide/index.html)。

### 4.8 编译运行

编译infer_cpp目录，在当前目录./bin/文件下生成可执行文件`edge_infer`.

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```

推理：

```bash 
#parameter: quantmodel batchsize
bash run.sh qint8_mixed_float16 1
``` 

### 4.9 精度验证

```
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：
```bash
Accuracy 0.9375310242716713
FP 0.2393961179007908
FN 0.09834052240594329
```

其他量化方式下 Ultra-Fast-Lane-Detecion 在 Tusimple 数据集中精度如下：

| Model     | Precision            | Batch_Size | Accuracy       | FP        | FN        |
| --------- | -------------------- | ---------- | -------------- | --------- | --------- |
| Tusimple  | force_float32        | 1          | 0.95767        | 0.19203   | 0.04109   |
| Tusimple  | force_float16        | 1          | 0.95767        | 0.1919    | 0.04097   |
| Tusimple  | qint16_mixed_float32 | 1          | 0.94511        | 0.21711   | 0.07129   |
| Tusimple  | qint16_mixed_float16 | 1          | 0.94516        | 0.21693   | 0.07111   |
| Tusimple  | qint8_mixed_float32  | 1          | 0.93781        | 0.23912   | 0.09813   |
| Tusimple  | qint8_mixed_float16  | 1          | 0.93753        | 0.23939   | 0.09834   |

### 4.10 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $RPOJ_ROOT_PATH/data/models/tusimple_qint8_mixed_float16_1.mm  --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

## 5 高级说明

### 5.1 gen_model参数说明
PyTorch Ultra-Fast-Lane-Detecion 模型转换为 MagicMind 模型分成以下几步：
* 使用MagicMind Parser模块将torch.jit.trace生成的pt文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `pt_model`: 转换后pt的路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行目标检测任务。
* `output_model_path`: 保存MagicMind模型路径。
* `quant_mode`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `batch_size`: batch大小，当前默认只支持1batch。

### 5.2 infer_cpp代码说明

本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的车道线检测程序(图像预处理=>推理=>后处理)。相关代码存放在infer_cpp目录下可供参考。其中程序主要由以下内容构成:

* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_precess.hpp, pre_precess.cpp: 前处理。
* post_precess.hpp, post_precess.cpp: 后处理

## 6 免责声明 

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* Tusimple 数据集下载链接：https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip
* Tusimple 标签下载链接：https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json
* Ultra-Fast-Lane-Detection 模型下载链接：https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/INSTALL.md
* Ultra-Fast-Lane-Detection GITHUB下载链接：https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
