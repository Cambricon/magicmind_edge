# yolov5_pytorch

MagicMind是面向寒武纪MLU的推理加速引擎。MagicMind能将 Tensorflow, PyTorch, ONNX 等训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。这份仓库探讨如何使用将yolov5从PyTorch转换为MagicMind模型，进而部署在寒武纪CE3226板卡上。

## 目录

* [模型概述](#1-模型概述)
* [前提条件](#2-前提条件)
* [快速使用](#3-快速使用)
  * [环境准备](#31-环境准备)
  * [下载仓库](#32-下载仓库)
  * [下载数据集模型](#33-下载数据集模型)
  * [目录结构](#34-目录结构)
  * [模型转换](#35-模型转换)
  * [生成MagicMind模型](#36-生成magicmind模型)
  * [编译运行](#37-编译运行)
  * [精度验证](#38-精度验证)
  * [MM_RUN_BENCHMARK性能结果](#39-mm_run_benchmark性能结果)
  * [一键运行](#310-一键运行)
* [高级说明](#4-高级说明)
  * [gen_model代码解释](#41-gen_model参数说明)
  * [infer_cpp代码解释](#42-infer_cpp代码说明)
* [免责声明](#5-免责声明)
* [Release Notes](#6-release-notes)

## 1 模型概述

 本例使用的yolov5实现来自github开源项目https://github.com/ultralytics/yolov5 中的v6.1版本。下面将展示如何将该项目中PyTorch实现的yolov5模型转换为MagicMind的模型。

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
cd magicmind_edge/buildin/cv/detection/yolov5_v6.1_pytorch
```

开始运行代码前需要先执行以下命令安装必要的库：
```bash
pip install -r requirements.txt
```
运行前，请检查以下路径：
```
NEUWARE_HOME=/usr/local/neuware
MAGICMIND_EDGE="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" \
COCO_DATASETS_PATH=$MAGICMIND_EDGE/datasets/coco
PROJ_ROOT_PATH=$MAGICMIND_EDGE/buildin/cv/detection/yolov5_v6.1_pytorch
UTILS_PATH=$MAGICMIND_EDGE/buildin/cv/utils
THIRD_PARTY=$MAGICMIND_EDGE/buildin/cv/3rdparty
```

### 3.3 下载数据集模型
```bash
# 下载数据集
cd $COCO_DATASETS_PATH
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip -o val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o annotations_trainval2017.zip

# 下载权重文件
wget -p $PROJ_ROOT_PATH/models -c https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt

# 下载yolov5源码
cd $PROJ_ROOT_PATH/export_model
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 切换到v6.1版本
git checkout v6.1
```
### 3.4 目录结构

```
.
|-- README.md
|-- benchmark
|-- gen_model
|-- export_model
|-- infer_cpp
|-- data
|-- requirements.txt
`-- run.sh
```
目录结构说明：
* benchmark: 提供mm_run测试脚本，用于测试该模型在不同输入规模、不同数据精度、不同硬件设备下的性能；

* gen_model: 主要涉及模型固化,量化和转为mm engine过程，要求内部能够一键执行完成该模块完整功能；

* export_model: 为适配不同模型算法适用于MagicMind框架运行，针对源码进行的patch修订；

* infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；一般图像类及前后处理不复杂的网络，建议要有c++的推理；

* data: 用于暂存测试结果，保存模型等;

* requirement.txt: 相关依赖；

* run.sh: 顶层一键执行脚本，串联各个部分作为整个sample的一键运行脚本；


### 3.5 模型转换

为了提高性能，需要将原yolov5模型中的detect层后处理去掉,通过torch.jit.trace 保存成固化pt模型,最后在生成MagicMind Model时添加yolov5后处理算子。

1. 将https://github.com/ultralytics/yolov5/blob/v6.1/models/yolo.py 中的Detect层截断。以Detect层的3个输入作为网络的输出，自定义yolov5_detection_output大算子运行于MLU中，加速运行效果。

```bash
cd yolov5 
cp ../patch/yolov5_v6.1_pytorch.patch ./
git apply yolov5_v6.1_pytorch.patch
```
2. 由于MagicMind最高支持pytorch 1.6.0版本，此版本没有SiLU函数，所以要在环境中修改代码如下：

```vim /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py```

```
#添加如下函数定义
class SiLU(Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
```
```vim /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py```

```
#添加声明
from .activation import SiLU
__all__ = [ *, 'SiLU']
```

3. jit.trace导出模型，模型默认保存在**models**文件夹下
```bash
cd yolov5
python export.py --weights $PROJ_ROOT_PATH/data/models/yolov5m.pt --include torchscript
```

### 3.6 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
bash run.sh qint8_mixed_float16 1 0.001 0.6
```
或手动执行以下命令:
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

结果默认保存在在**models**文件夹中。有关更多生成MagicMind模型参数相关设置,请参考**//TODO

### 3.7 编译运行

编译infer_cpp目录，在当前目录./bin/文件下生成可执行文件`edge_infer`.

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```

推理：

```bash 
bash $PROJ_ROOT_PATH/infer_cpp/bin/edge_infer \
      --magicmind_model $PROJ_ROOT_PATH/data/models/yolov5m_qint8_mixed_float16_1.mm \
      --image_dir $COCO_DATASETS_PATH/coco/val2017/ \
      --save true \
      --output_dir $PROJ_ROOT_PATH/data/images \
      --label_path $UTILS_PATH/coco.names \
      --coco_result $PROJ_ROOT_PATH/data/result.json
```  
参数解析：
      --magicmind_model 输入模型路径
      --image_dir 测试数据路径
      --save 是否保存检测图像结果，默认为false
      --output_dir 检测结果保存路径
      --label_path coco.names路径
      --coco_result 保存json精度结果

### 3.8 精度验证

```
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.425
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.159
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.353
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.240
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.266
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
```
### 3.9 MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $RPOJ_ROOT_PATH/data/models/yolov5m_qint8_mixed_float16_1.mm --devices $DEV_ID --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

### 3.10 一键运行
以上3.2~3.9的步骤，均可以通过```bash run.sh``` 实现一键运行。

**注:** 
* 1. 执行一键运行脚本,需用户确保已下载好模型和数据集; 
* 2. 非云平台用户在本地测试时,可将$MAGICMIND_EDGE路径下**datasets_test**文件软连接为**datasets**,即```ln -s datasets_test datasets```;
* 3. **datasets_test**文件为数据集路径,仅包含少量测试用例/图片,使用于用户进行功能验证,无法确保精度结果; 
* 4. 非云平台用户若需测试完整精度结果,需将本地完整的数据集路径软连接到$MAGICMIND_EDGE/datasets路径下; 

## 4 高级说明

### 4.1 gen_model参数说明
PyTorch yolov5模型转换为MagicMind yolov5模型分成以下几步：
* 使用MagicMind Parser模块将torch.jit.trace生成的pt文件解析为MagicMind网络结构。
* 模型量化。
* 添加yolov5后处理算子。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `pt_model`: 转换后pt的路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行目标检测任务。
* `output_model_path`: 保存MagicMind模型路径。
* `quant_mode`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `shape_mutable`: 是否生成可变batch_size的MagicMind模型。

### 4.2 infer_cpp代码说明

本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的yolov5目标检测程序(图像预处理=>推理=>后处理)。相关代码存放在infer_cpp目录下可供参考。其中程序主要由以下内容构成:

* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_precess.hpp, pre_precess.cpp: 前处理。
* post_precess.hpp, post_precess.cpp: 后处理

## 5 免责声明 

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* COCO VAL2017 数据集下载链接：http://images.cocodataset.org/zips/val2017.zip
* COCO VAL2017 标签下载链接：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
* YOLOV5M 模型下载链接：https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt
* YOLOV5 GITHUB下载链接：https://github.com/ultralytics/yolov5.git

## 6 Release Notes
@TODO 
