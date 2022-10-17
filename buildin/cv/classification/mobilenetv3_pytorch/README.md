# mobilenetv3_pytorch

MagicMind是面向寒武纪 MLU 的推理加速引擎。MagicMind能将 Tensorflow, PyTorch, ONNX 等训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。本sample探讨如何使用将mobilenetv3网络的pytorch模型转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

## 目录
* [模型概述](#1-模型概述)
* [前提条件](#2-前提条件)
* [快速使用](#3-快速使用)
  * [环境准备](#31-环境准备)
  * [下载仓库](#32-下载仓库)
  * [下载数据集模型](#33-下载数据集模型)
  * [模行转换](#34-模型转换)
  * [生成MagicMind模型](#35-生成magicmind模型)
  * [编译运行](#36-编译运行)
  * [精度验证](#37-精度验证)
  * [一键运行](#38-一键运行)
* [高级说明](#4.高级说明)
  * [gen_model代码解释](#41-gen_model代码解释)
  * [infer_cpp代码解释](#42-infer_cpp代码解释)
* [免责声明](#5-免责声明)
* [Release Notes](#6-release-notes)

## 1 模型概述

本例使用的mobilenetv3实现来自github开源项目https://github.com/kuan-wang/pytorch-mobilenet-v3

下面将展示如何将该项目中pytorch实现的mobilenetv3模型转换为MagicMind的模型。

## 2 前提条件

* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本 MLU370 S4或 MLU370 X4，并安装好驱动(>=v4.20.6)；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 3.快速使用
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
cd magicmind_edge/buildin/cv/classification/mobilenetv3_pytorch 
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
```
if [ ! -f $PROJ_ROOT_PATH/data/models/mobilenetv3_small_67.4.pth.tar ];then
  echo "downloading mobilenetv3_small_67.4.pth.tar"
  gdown -c https://drive.google.com/uc?id=1lCsN3kWXAu8C30bQrD2JTZ7S2v4yt23C -O $PROJ_ROOT_PATH/data/models/mobilenetv3_small_67.4.pth.tar 
fi
```
### 3.4 模型转换

将网络原始模型通过torch.jit.trace固化.

```
cd $PROJ_ROOT_PATH/export_model 
./run.sh
```

### 3.5 生成MagicMind模型

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

### 3.6 编译运行

进入infer_cpp目录，在当前目录编译生成可执行文件edge_infer:

```
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```
推理:

```
cd $PROJ_ROOT_PATH/infer_cpp
bash ./bin/edge_infer \
        --magicmind_model $PROJ_ROOT_PATH/data/models/mobilenetv3_${QUANT_MODE}_${BATCH_SIZE}.mm \
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
top 1 accuracy: 0.642940
top 2 accuracy: 0.756400
top 3 accuracy: 0.805720
top 4 accuracy: 0.833240
top 5 accuracy: 0.851860
```

### 3.8 一键运行

步骤3.2至3.7可通过脚本./run.sh来实现一键执行.

## 4 高级说明
### 4.1 gen_model细节说明
pytorch mobilenetv3模型转换为MagicMind mobilenetv3模型分成以下几步：
* 使用MagicMind Parser模块将pytorch文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `pytorch_model`: mobilenetv3 pytorch的权重路径。
* `output_model_path`: 保存MagicMind模型路径。
* `image_dir`: 输入图像目录1。
* `quant_mode`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `batch_size`: batch大小，默认为1。
* `remote_addres:`(可选)远端IP地址，是否采用远端量化方式y。

### 4.2 infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的视频检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的mobilenetv3目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:
* infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_process.hpp, pre_process.cpp 前处理

参数说明:
* magicmind_model: MagicMind模型路径。
* image_dir: 数据集路径
* output_dir:推理输出-画框图像路径

## 5 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* mobilenetv3 pth模型下载链接：https://drive.google.com/uc?id=1lCsN3kWXAu8C30bQrD2JTZ7S2v4yt23C

## 6 Release Notes
@TODO
