# inceptionv3_caffe

 MagicMind是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将Tensorflow, PyTorch, Caffe等训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。这份仓库探讨如何将Caffe分类网络inceptionv3转换为MagicMind模型，进而部署在寒武纪CE3226板卡上。

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
  * [gen_model代码解释](#41-gen_model代码解释)
  * [infer_cpp代码解释](#42-infer_cpp代码解释)
* [免责声明](#5-免责声明)
* [Release Notes](#6-release-notes)

## 1 模型概述

 本例使用的inceptionv3型来自github开源项目 https://github.com/soeaver/caffe-model/tree/master/cls/inception。
 
 下面展示如何将该项目中Caffe框架下inceptionv3模型（prototxt,caffemodel）转换为MagicMind的模型。

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

* 下载仓库
```bash
git clone git@gitee.com:cambricon/magicmind_edge.git
cd magicmind_edge/buildin/cv/classification/inceptionv3_caffe
```

运行前，请检查以下路径，或执行```source env.sh```

```
NEUWARE_HOME=/usr/local/neuware \
PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" \
MAGICMIND_EDGE="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" \
IMAGENET_DATASETS_PATH=$MAGICMIND_EDGE/datasets/imagenet \
UTILS_PATH=$MAGICMIND_EDGE/buildin/cv/utils \
THIRD_PARTY=$MAGICMIND_EDGE/buildin/cv/3rdparty \
MM_RUN_PATH=$NEUWARE_HOME/bin 
```

### 3.3 下载数据集模型

* 下载测试数据集
```bash
cd $IMAGENET_DATASETS_PATH
# 请自行从https://image-net.org/challenges/LSVRC/下载LSVRC_2012验证集并解压至$IMAGENET_DATASETS_PATH/ILSVRC2012_img_val目录
```

* 下载权重

请用户从下述github链接下载模型权重至$PROJ_ROOT_PATH/data/models.

[inception-v3.caffemodel](https://github.com/soeaver/caffe-model/tree/master/cls#performance-on-imagenet-validation)

[deploy_inception-v3.prototxt](https://github.com/soeaver/caffe-model/blob/master/cls/inception/deploy_inception-v3.prototxt)

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

### 3.5 生成MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_model batch_size
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

结果默认保存在在$PROJ_ROOT_PATH/data/models文件夹中。
有关更多生成MagicMind模型参数相关设置,请参考//TODO

### 3.6 编译运行

进入infer_cpp目录，在当前目录编译生成可执行文件`edge_infer`:

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```

推理：

```bash 
bash ./bin/edge_infer \ #或 bin/host_infer在host端进行推理验证
      --magicmind_model $PROJ_ROOT_PATH/data/models/inceptionv3_qint8_mixed_float16_1.mm \
      --image_dir $IMAGENET_DATASETS_PATH/ \
      --output_dir $PROJ_ROOT_PATH/data/images 
#参数解析：
      --magicmind_model 输入模型路径
      --image_dir 测试数据路径
      --output_dir 检测结果保存路径
```  

### 3.7 精度验证

```bash
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```

结果：
```bash
top 1 accuracy: 0.729000
top 2 accuracy: 0.840000
top 3 accuracy: 0.878000
top 4 accuracy: 0.902000
top 5 accuracy: 0.916000
```
### 3.8MM_RUN_BENCHMARK性能结果

本仓库通过寒武纪提供的MagicMind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/inceptionv3_qint8_mixed_float16_1.mm --threads 1 --iterations 1000
```
或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1
```

### 3.9 一键运行

以上3.2~3.9的步骤，均可以通过```bash run.sh``` 实现一键运行。

**注:** 
* 1. 执行一键运行脚本,需用户确保已下载好模型和数据集; 
* 2. 非云平台用户在本地测试时,可将$MAGICMIND_EDGE路径下**datasets_test**文件软连接为**datasets**,即```ln -s datasets_test datasets```;
* 3. **datasets_test**文件为数据集路径,仅包含少量测试用例/图片,使用于用户进行功能验证,无法确保精度结果; 
* 4. 非云平台用户若需测试完整精度结果,需将本地完整的数据集路径软连接到$MAGICMIND_EDGE/datasets路径下;

## 4 高级说明

### 4.1 gen_model代码解释
caffe 模型转换为MagicMind，其流程较为简单易操作，可直接采用生成工具（脚本）对原始caffe框架的prototxt和caffemodel进行操作。

gen_model.py参数说明:
* `prototxt`: caffe框架下的模型结构文件。
* `caffe_model`: caffe框架下的权重文件。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行目标检测任务。
* `output_model_path`: 保存MagicMind模型路径。
* `quant_mode`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `batch_size`: batch大小，默认为1。
* `remote_addres`: (可选)远端IP地址，是否采用远端量化方式。


### 4.2 infer_cpp代码解释
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的inceptionv3分类程序(图像预处理=>推理=>后处理)。相关代码存放在infer_cpp目录下可供参考。其中程序主要由以下内容构成:

* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_precess.hpp, pre_precess.cpp: 前处理。

## 5 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* inceptionv3 prototxt GITHUB下载链接:https://github.com/soeaver/caffe-model/blob/master/cls/inception/deploy_inception-v3.prototxt
* inceptionv3 caffemodel GITHUB下载链接:https://github.com/soeaver/caffe-model/tree/master/cls#performance-on-imagenet-validation
* Imagenet 数据集下载链接:https://image-net.org/challenges/LSVRC/
## 6 Release Notes
