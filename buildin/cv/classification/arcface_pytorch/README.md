# arcface_pytorch

 MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 Tensorflow, PyTorch, Caffe 等训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。这份仓库探讨如何将Pytorch人脸识别网络arcface转换为MagicMind模型，进而部署在寒武纪CE3226板卡上。

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
  * [gen_model代码解释](#41-gen_model代码解释)
  * [infer_cpp代码解释](#42-infer_cpp代码解释)
* [免责声明](#5-免责声明)
* [Release Notes](#6-release-notes)

## 1 模型概述

 本例使用的arcface模型来自github开源项目 https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch。
 
 下面展示如何将该项目中Pytorch框架下arcface模型转换为MagicMind的模型。

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
cd magicmind_edge/buildin/cv/classification/arcface_pytorch
```

运行前，请检查以下路径，或执行```source env.sh```

```
NEUWARE_HOME=/usr/local/neuware \
PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" \
MAGICMIND_EDGE="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" \
IJB_DATASETS_PATH=$MAGICMIND_EDGE/datasets/IJB \
UTILS_PATH=$MAGICMIND_EDGE/buildin/cv/utils \
THIRD_PARTY=$MAGICMIND_EDGE/buildin/cv/3rdparty \
MM_RUN_PATH=$NEUWARE_HOME/bin 
```

### 3.3 下载数据集,模型
* 数据集

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

请将ms1mv3_arcface_r100_fp16目录中的backbone.pth 下载至```$PROJ_ROOT_PATH/data/models/```目录.

### 3.4 目录结构

```
.
|-- README.md
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

* export_model: 用于生成torch.jit.trace模型;

* infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；一般图像类及前后处理不复杂的网络，建议要有c++的推理；

* run.sh: 顶层一键执行脚本，串联各个部分作为整个sample的一键运行脚本；

### 3.5 模型转换

```
cd $PROJ_ROOT_PATH/export_model
#param: quant_mode batch_size
bash run.sh qint8_mixed_float16 1
```

### 3.6 编译生成MagicMind模型

```
cd $PROJ_ROOT_PATH/gen_model
#param: quant_mode batch_size
bash run.sh qint8_mixed_float16 1
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

注：
1. 本实例支持batch_size设置功能，已测试最大规模为256。
2. 结果默认保存在$PROJ_ROOT_PATH/data/models文件夹。
3. 有关更多生成MagicMind模型参数相关设置,请参考//TODO

### 3.7 编译运行

进入infer_cpp目录，在当前目录编译生成可执行文件`edge_infer`:

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```

推理：

```bash 
bash ./bin/edge_infer \ #或 bin/host_infer在host端进行推理验证
    --magicmind_model $PROJ_ROOT_PATH/data/models/arcface_qint8_mixed_float16_1.mm \
    --image_dir $IJB_DATASETS_PATH/IJBC/loose_crop \
    --image_list $IJB_DATASETS_PATH/IJBC/meta/ijbc_name_5pts_score.txt \
    --save_img true \
    --output_dir $PROJ_ROOT_PATH/data/images  
#参数解析：
      --magicmind_model 输入模型路径
      --image_dir 测试图片路径
      --image_list 测试图片file_list
      --save_img 是否保存检测结果
      --output_dir 检测结果保存路径
```  

### 3.8 精度验证

```
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：
```bash
IJB-C(1E-5)          IJB—C(1E-4)
{'1e-5': '94.90208', '1e-4': '96.59968'}
```
### 3.9 MM_RUN_BENCHMARK性能结果

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

### 3.10 一键运行

以上3.2~3.9的步骤，均可以通过```bash run.sh``` 实现一键运行。

**注:** 
* 1. 执行一键运行脚本,需用户确保已下载好模型和数据集; 
* 2. 非云平台用户在本地测试时,可将$MAGICMIND_EDGE路径下**datasets_test**文件软连接为**datasets**,即```ln -s datasets_test datasets```;
* 3. **datasets_test**文件为数据集路径,仅包含少量测试用例/图片,使用于用户进行功能验证,无法确保精度结果; 
* 4. 非云平台用户若需测试完整精度结果,需将本地完整的数据集路径软连接到$MAGICMIND_EDGE/datasets路径下; 

## 4 高级说明

### 4.1 gen_model代码解释
Pytorch arcface 模型转换为MagicMind，其流程主要分为以下两步:

* 将原始pth模型通过torch.jit.trace生成固化模型(*.pt).
* 使用MagicMind Parser模块将torch.jit.trace生成的pt文件解析为MagicMind网络结构.
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

gen_model.py参数说明:
* `pt_model`: 转换后pt的路径。
* `image_dir`: 输入图像file_list,保存输入图像的路径。
* `output_model_path`: 保存MagicMind模型路径。
* `quant_mode`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `batch_size`: batch大小，默认为1。
* `remote_addres`: (可选)远端IP地址，是否采用远端量化方式。


### 4.2 infer_cpp代码解释
概述:
本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的arcface人脸识别程序(图像预处理=>推理=>后处理)。相关代码存放在infer_cpp目录下可供参考。其中程序主要由以下内容构成:

* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_precess.hpp, pre_precess.cpp: 前处理。

## 5 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* IJB数据集下载链接:https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view
* IJB数据集下载链接:https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg
* arcface GITHUB下载链接:https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
* arcface backborn模型下载链接:https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC

## 6 Release Notes
