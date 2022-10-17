# centernet_pytorch

MagicMind是面向寒武纪 MLU 的推理加速引擎。MagicMind能将 Tensorflow, PyTorch, ONNX 等训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。本sample探讨如何使用将CenterNet网络的PyTorch实现转换为MagicMind模型，进而部署在寒武纪CE3226板卡上。

## 目录

* [模型概述](#1-模型概述)
* [前提条件](#2-前提条件)
* [快速使用](#3-快速使用)
  * [环境准备](#31-环境准备)
  * [下载仓库](#32-下载仓库)
  * [下载数据集模型](#33-下载数据集模型)
  * [模型转换](#34-模型转换)
  * [编译MagicMind模型](#35-编译magicmind模型)
  * [编译运行](#36-编译运行)
  * [精度验证](#3.7-精度验证)
  * [一键运行](#3.8一键运行)
* [高级说明](#4-高级说明)
  * [export_model细节说明](#41-export_model细节说明)
  * [gen_model细节说明](#42-gen_model细节说明)
  * [infer_cpp细节说明](#43-infer_cpp细节说明)
* [免责声明](#5-免责声明) 
* [Release Notes](#6-release-notes)

## 1 模型概述
本例使用的CenterNet实现来自github开源项目https://github.com/xingyizhou/CenterNet/tree/2b7692c377c6686fb35e473dac2de6105eed62c6。

下面将展示如何将该项目中PyTorch实现的CenterNet网络转换为MagicMind的模型。由于MagicMind暂不支持DCN算子，本例可使用的Backbone有Hourglass以及DLAV0，下面使用DLAV0作为backbone进行部署。


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
cd magicmind_edge/buildin/cv/detection/centernet_pytorch/
```

在开始运行代码前需要执行以下命令安装必要的库：

```bash
pip install -r requirement.txt
```

在开始运行代码前需要先检查env.sh里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集模型
```bash
cd $PROJ_ROOT_PATH/export_model 
./get_datasets_and_models.sh
```

### 3.4 模型转换
```bash
cd $PROJ_ROOT_PATH/export_model 
#params:batch_size
./run.sh 1
```

### 3.5 编译MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#params:quant_model batch_size
./run.sh qint8_mixed_float16 1 
```

结果：

```bash 
Generate model done, model save to /*/magicmind_edge/buildin/cv/detection/centernet_pytorch/data/models/centernet_qint8_mixed_float16_1.mm
```

### 3.6 编译运行

编译infer_cpp目录，在当前目录输出x86可执行文件`infer`:
```bash
cd $PROJ_ROOT_PATH/infer_cpp
./build.sh
```

推理：
```bash 
bash run.sh qint8_mixed_float16 1 100
```

### 3.7 精度验证

```bash
python $UTILS_PATH/compute_coco_mAP.py --file_list $DATASETS_PATH/file_list_5000.txt \
                                       --result_dir $PROJ_ROOT_PATH/data/images/infer_cpp_output_dir_qint8_mixed_float16_true_1 \
                                       --ann_dir $DATASETS_PATH \
                                       --data_type val2017 \
                                       --json_name $PROJ_ROOT_PATH/data/json/centernet_pytorch_model_qint8_mixed_float16_true_1 \
                                       --img_num 10
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

### 3.8 一键运行
以上3.3 ~ 3.7的步骤也可以通过运行./run.sh来实现一键执行

## 4 高级说明
### 4.1 export_model细节说明
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

### 4.2 gen_model细节说明
PyTorch CenterNet网络转换为MagicMind模型可分为以下几个步骤:
1.通过torch.jit.trace生成pt文件。
2.使用MagicMind Parser模块将torch.jit.trace生成的pt文件解析为MagicMind网络结构。
3.模型量化(可选)。
4.使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `pt_model`: 转换后pt的路径。
* `output_model`: 保存MagicMind模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行目标检测任务。
* `quant_mode`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `shape_mutable`: 是否生成可变batch_size的MagicMind模型。
* `batch_size`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。

### 4.3 infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的centernet目标检测(图像预处理=>推理=>后处理)。其中程序主要由以下内容构成:
* infer.hpp, infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。
* pre_precess.hpp, pre_precess.cpp: 前处理。
* post_precess.hpp, post_precess.cpp: 后处理。

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

## 5 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* COCO VAL2017 数据集下载链接：http://images.cocodataset.org/zips/val2017.zip
* COCO VAL2017 标签下载链接：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
* Centernet权重文件下载链接：https://drive.google.com/uc?id=18yBxWOlhTo32_swSug_HM4q3BeWgxp_N
* CenterNet实现源码下载链接：https://github.com/xingyizhou/CenterNet/archive/2b7692c377c6686fb35e473dac2de6105eed62c6.zip

## 6 Release Notes
@TODO
