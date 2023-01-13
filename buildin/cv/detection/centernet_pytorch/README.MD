# centernet_pytorch

MagicMind 是面向寒武纪 MLU 推理加速引擎。MagicMind 能将 AI 框架(Tensorflow, PyTorch, Caffe, ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 CenterNet 从 PyTorch 转换为 MagicMind 模型，进而部署在寒武纪CE3226板卡上。

## 目录

- [centernet_pytorch](#centernet_pytorch)
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
    - [4.6 模型转换](#46-模型转换)
    - [4.7 生成MagicMind模型](#47-生成magicmind模型)
    - [4.8 编译运行](#48-编译运行)
    - [4.9 精度验证](#49-精度验证)
    - [4.10 MM_RUN_BENCHMARK性能结果](#410-mm_run_benchmark性能结果)
  - [5 高级说明](#5-高级说明)
    - [5.1 export_model细节说明](#51-export_model细节说明)
    - [5.2 gen_model细节说明](#52-gen_model细节说明)
    - [5.3 infer_cpp细节说明](#53-infer_cpp细节说明)
  - [6 免责声明](#6-免责声明)

## 1 模型概述
本例使用的CenterNet实现来自github开源项目https://github.com/xingyizhou/CenterNet/tree/2b7692c377c6686fb35e473dac2de6105eed62c6。

下面将展示如何将该项目中PyTorch实现的CenterNet网络转换为MagicMind的模型。由于MagicMind暂不支持DCN算子，本例可使用的Backbone有Hourglass以及DLAV0，下面使用DLAV0作为backbone进行部署。


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
|-- requirements.txt
|-- run.sh
```
目录结构说明：
* benchmark: 提供mm_run测试脚本，用于测试该模型在不同输入规模、不同数据精度、不同硬件设备下的性能；同时提供精度验证功能；
* data: 用于暂存测试结果，保存模型等;
* gen_model: 主要涉及模型量化和转为mm engine过程，要求内部能够一键执行完成该模块完整功能；
* env.sh: 环境变量，需要在模型运行之前source env.sh；
* export_model: 用于下载模式和数据集；
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
cd magicmind_edge/buildin/cv/detection/centernet_pytorch/
```
### 4.3 设置环境变量

在开始运行代码前需要执行以下命令安装必要的库：

```bash
pip install -r requirements.txt
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
以下4.5 ~ 4.10 的步骤也可以通过运行./run.sh来实现一键执行
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
./get_datasets_and_models.sh
```

### 4.6 模型转换
```bash
cd $PROJ_ROOT_PATH/export_model 
#params:batch_size
./run.sh 1
```

### 4.7 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#params:quant_model batch_size
./run.sh qint8_mixed_float16 1 
```

结果：

```bash 
Generate model done, model save to /*/magicmind_edge/buildin/cv/detection/centernet_pytorch/data/models/centernet_qint8_mixed_float16_1.mm
```

### 4.8 编译运行

编译infer_cpp目录，在当前目录./bin/文件下生成可执行文件`edge_infer`.

```bash
cd $PROJ_ROOT_PATH/infer_cpp
./build.sh
```

推理：
```bash 
#parameter: quantmodel batchsize imagenum
bash run.sh qint8_mixed_float16 1 100
```

### 4.9 精度验证

```bash
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.495
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.222
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.690
```
### 4.10 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/centernet_qint8_mixed_float16_1.mm --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

## 5 高级说明
### 5.1 export_model细节说明
向MagicMind导入PyTorch模型首先需要先使用torch.jit.trace生成PyTorch模型文件。 

首先，执行以下代码块修改CenterNet源码。将CenterNet后处理过程中Heatmap的池化操作放入网络中执行，这样在后面的检测程序中，池化操作将包含在MagicMind模型中，并运行在MLU上。
加入此项修改后，网络变为4个输出: hm_max(经过池化后的heatmap)，heatmap，wh(检测框宽高), reg(回归中心点偏移量)。

参考https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/decode.py 中_nms函数实现。

```bash
cd $PROJ_ROOT_PATH/export_model
# 这个patch文件只修改了dlav0作为backbone的实现，若需使用其它backbone，可参照centernet.diff文件中的修改内容。
patch CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/networks/dlav0.py < centernet_dlav0.diff
# 注释掉DCN相关内容
patch CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/model.py < centernet_model.diff
# 使用$PROJ_ROOT_PATH/export_model/export.py jit.trace导出模型。
python $PROJ_ROOT_PATH/export_model/export.py --model_weight $MODEL_PATH/ctdet_coco_dlav0_1x.pth \
    					      --input_width 512 \
    					      --input_height 512 \
    					      --batch_size 1 \
    					      --traced_pt $PROJ_ROOT_PATH/data/models/ctdet_coco_dlav0_1x_traced_1bs.pt
```

### 5.2 gen_model细节说明
PyTorch CenterNet网络转换为MagicMind模型可分为以下几个步骤:
1.通过torch.jit.trace生成pt文件。
2.使用MagicMind Parser模块将torch.jit.trace生成的pt文件解析为MagicMind网络结构。
3.模型量化(可选)。
4.使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `pt_model`: 转换后pt的路径。
* `output_model`: 保存MagicMind模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行目标检测任务。
* `quant_mode`: 量化模式，如force_float32，force_float16，qint8_mixed_float16。
* `shape_mutable`: 是否生成可变batch_size的MagicMind模型。
* `batch_size`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。

### 5.3 infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的centernet目标检测(图像预处理=>推理=>后处理)。其中程序主要由以下内容构成:
* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_process.hpp, pre_process.cpp: 前处理。
* post_process.hpp, post_process.cpp: 后处理。

参数说明:
* `magicmind_model`: MagicMind模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行目标检测任务。
* `image_num`: 输入图像数量。
* `file_list`: 数据集文件列表文件。
* `label_path`: 标签文件路径。
* `max_bbox_num`:最大毛框数量。
* `confidence_thresholds`: confidence_thresh，检测框得分阈值。
* `output_dir`: 根据检测结果进行渲染后的图像或COCO API风格检测结果文件保存路径。
* `save_img`: 结果可视化。若指定为true，则保存渲染后的图像，默认为false。
* `batch`: 推理时，输入tensor的维度

## 6 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* COCO VAL2017 数据集下载链接：http://images.cocodataset.org/zips/val2017.zip
* COCO VAL2017 标签下载链接：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
* Centernet权重文件下载链接：https://drive.google.com/uc?id=18yBxWOlhTo32_swSug_HM4q3BeWgxp_N
* CenterNet实现源码下载链接：https://github.com/xingyizhou/CenterNet/archive/2b7692c377c6686fb35e473dac2de6105eed62c6.zip

