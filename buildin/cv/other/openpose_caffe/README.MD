# openpose_caffe

MagicMind 是面向寒武纪 MLU 推理加速引擎。MagicMind 能将 AI 框架(Tensorflow, PyTorch, Caffe, ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 openpose 从 Caffe 转换为 MagicMind 模型，进而部署在寒武纪CE3226板卡上。支持 BODY_25 及 COCO 两个模型。

## 目录

- [openpose\_caffe](#openpose_caffe)
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
    - [5.1 gen\_model参数说明](#51-gen_model参数说明)
    - [5.2 infer\_cpp 高级说明](#52-infer_cpp-高级说明)
  - [6 免责声明](#6-免责声明)

## 1 模型概述

本例使用的 openpose 实现来自github开源项目https://github.com/CMU-Perceptual-Computing-Lab/openpose。下面将展示如何将该项目中Caffe实现的openpose模型转换为MagicMind的模型。

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
* export_model:用户下载数据集与模型；
* infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；
* requirements.txt: 依赖包；
* run.sh: 顶层一键执行脚本，串联各个部分作为整个sample的一键运行脚本；

## 4 快速使用

### 4.1 环境准备

请移至[主页面 README.md](README.md)的`3.环境准备`

### 4.2 下载仓库

```bash
# 下载仓库
git clone http://gitlab.software.cambricon.com/neuware/software/ae/ecosystem/modelzoo/magicmind_edge.git
cd magicmind_edge/buildin/cv/other/openpose_caffe
```
### 4.3 设置环境变量

开始运行代码前需要先执行以下命令安装必要的库：
```bash
pip install -r requirement.txt
```

运行前，请检查以下路径，或执行```source env.sh```：

```bash
export PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export MAGICMIND_EDGE="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export COCO_DATASETS_PATH=$MAGICMIND_EDGE/datasets/coco
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

以下4.5~4.9的步骤，均可以通过：
```
bash run.sh
``` 
实现一键运行，该脚本包含数据集、模型的自动下载，模型转换与生成，推理代码编译与运行，性能精度测试流程。

**注:** 

一键运行脚本中包含模型与数据集的自动下载，部分数据集与模型由于网络、链接等原因无法自动下载时，需要用户根据README.MD中链接地址手动下载到指定位置。

下面对一键运行脚本中各个步骤进行逐步解析。

### 4.5 下载数据集和模型
```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 4.6 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode batch_size
bash run.sh qint8_mixed_float16 1
```
上述 run.sh 脚本中默认使用远程3226卡板进行模型生成，即设置了**REMOTE_IP**环境变量。

当本地为 MLU370 环境时，可直接执行以下命令进行模型生成:

```
# BODY_25
python $PROJ_ROOT_PATH/gen_model/gen_model.py --prototxt $MODEL_PATH/pose_deploy.prototxt \
	                                            --caffemodel $MODEL_PATH/pose_iter_584000.caffemodel \
                                              --output_model $MODEL_PATH/pose_body25_${PRECISION}_${BATCH_SIZE} \
						                                  --batchsize $BATCH_SIZE \
						                                  --precision $PRECISION \
                                              --calibrate_list $PROJ_ROOT_PATH/gen_model/calibrate_list.txt

# COCO
python $PROJ_ROOT_PATH/gen_model/gen_model.py --prototxt $MODEL_PATH/pose_deploy_linevec.prototxt \
	                                            --caffemodel $MODEL_PATH/pose_iter_440000.caffemodel \
                                              --output_model $MODEL_PATH/pose_coco_${PRECISION}_${BATCH_SIZE} \
						                                  --batchsize $BATCH_SIZE \
						                                  --precision $PRECISION \
                                              --calibrate_list $PROJ_ROOT_PATH/gen_model/calibrate_list.txt

```

生成的Magicmind模型默认保存在在$PROJ_ROOT_PATH/data/models文件夹中。

有关更多生成MagicMind模型参数相关设置,请参考[寒武纪Magicmind用户手册](https://www.cambricon.com/docs/sdk_1.7.0/magicmind_0.13.1/user_guide/index.html)。

### 4.7 编译运行

编译infer_cpp目录，在当前目录./bin/文件下生成可执行文件`edge_infer`.

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```

推理：

```bash 
#parameter: quant mode batchsize
bash run.sh qint8_mixed_float16 1
``` 

### 4.8 精度验证

```
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.541
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.774
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.584
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.491
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.790
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.706
......
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.505
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.743
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.535
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.557
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.770
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.689

```

### 4.9 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $RPOJ_ROOT_PATH/data/models/pose_body25_qint8_mixed_float16_1.mm --devices $DEV_ID --threads 1 --iterations 1000

mm_run --magicmind_model $RPOJ_ROOT_PATH/data/models/pose_coco_qint8_mixed_float16_1.mm --devices $DEV_ID --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

## 5 高级说明

### 5.1 gen_model参数说明
Caffe openpose 模型转换为 MagicMind openpose 模型分成以下几步：
* 使用 MagicMind Parser 模块将 caffe 文件解析为 MagicMind 网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:

* `prototxt`: caffe 的网络结构路径。
* `caffe_model`: caffe 的权重路径。
* `batch_size`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
* `input_width`: W。
* `input_height`: H。
* `output_model`: 保存 MagicMind 模型路径。
* `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
* `calibrate_list`: 输入图片文件列表，使用列表中文件做为输入进行模型量化。
* `device_id`: 设备号。

### 5.2 infer_cpp 高级说明

概述：
本例使用MagicMind C++ API编写了名为infer_cpp的模型推理程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的openpose人体姿态估计程序(图像预处理=>推理=>后处理)。相关代码存放在infer_cpp目录下可供参考。其中程序主要由以下内容构成:

* infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。
* pre_precess.hpp, pre_precess.cpp: 前处理。
* post_precess.hpp, post_precess.cpp: 后处理。

参数说明:

* `magicmind_model`: MagicMind 模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
* `image_list`: 数据集文件列表文件。
* `save_img`: 结果可视化。若指定为 true，则保存渲染后的图像，默认为 false。
* `output_dir`: 根据检测结果进行渲染后的图像或 COCO API 风格检测结果文件保存路径。
* `network`: 指定推理模型是 BODY_25 还是 COCO


## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* COCO VAL2017 数据集下载链接：http://images.cocodataset.org/zips/val2017.zip
* COCO VAL2017 标签下载链接：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
* BODY_25 prototxt 下载链接：https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt
* BODY_25 caffemodel 下载链接：http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel
* COCO prototxt 下载链接：https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
* COCO caffemodel 下载链接：http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
* openpose GITHUB 下载链接：https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
* rapidjson 代码下载链接：https://github.com/miloyip/rapidjson.git
