import os
import json
import numpy as np
import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType
from adapter_model import append_yolov7_detect
from calibrator import FixedCalibData

PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
COCO_DATASETS_PATH = os.environ.get("COCO_DATASETS_PATH")

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pt_model", type=str, default= str(PROJ_ROOT_PATH) + '/data/models/yolov7_traced.pt', help="modified yolov7 pt")
#parser.add_argument("-i", "--image_dir",  type=str, default="../../../../../datasets/coco/test10", help="coco2017 datasets")
parser.add_argument("-i", "--image_dir",  type=str, default= str(COCO_DATASETS_PATH)+ '/val2017', help="coco2017 datasets")
parser.add_argument("-o", "--output_model_path", type=str, default= str(PROJ_ROOT_PATH) + '/data/models/yolov7_qint8_mixed_float16_1.mm', help="save mm model to this path")
parser.add_argument("-q", "--quant_mode", type=str, default="qint8_mixed_float16", help="qint8_mixed_float16, qint8_mixed_float32, qint16_mixed_float16, qint16_mixed_float32, forced_float32, forced_float16")
parser.add_argument("-s", "--shape_mutable", type=str, default="false", help="whether the mm model is dynamic or static or not")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("-c", "--conf_thres", type=float, default=0.001, help="confidence_thresh")
parser.add_argument("-iou", "--iou_thres", type=float, default=0.65, help="nms_thresh")
parser.add_argument("-m", "--max_det", type=int, default=1000, help="limit_detections")
parser.add_argument("-r", "--remote_addres", type=str, default=os.environ.get('REMOTE_IP', None))

if __name__ == "__main__":
    args = parser.parse_args()
    
    network = mm.Network()
    pytorch_parser = Parser(mm.ModelKind.kPytorch)
    pytorch_parser.set_model_param("pytorch-input-dtypes", [DataType.FLOAT32])
    assert pytorch_parser.parse(network, args.pt_model).ok()
    
    network.get_input(0).set_dimension(mm.Dims((args.batch_size, 3, 640, 640)))
    network = append_yolov7_detect(network, args.conf_thres, args.iou_thres, args.max_det)
    
    config = mm.BuilderConfig()
   # config.parse_from_string('{"archs":["mtp_372"]}')
    config.parse_from_string('{"archs":["mtp_322.10"]}')
    config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}')
    config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}')
    config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}')
    
    ###static or dynamic model
    if args.shape_mutable == "true":
        config.parse_from_string('{"graph_shape_mutable": true}')
        config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 640, 640], "max": [64, 3, 640, 640]}}}')
    else:
        config.parse_from_string('{"graph_shape_mutable": false}')

    ###quantazation mode
    if args.quant_mode == "qint8_mixed_float16":
        assert config.parse_from_string('{"precision_config": {"precision_mode": "qint8_mixed_float16"}}').ok()
    elif args.quant_mode == "qint8_mixed_float32":
        assert config.parse_from_string('{"precision_config": {"precision_mode": "qint8_mixed_float32"}}').ok()
    elif args.quant_mode == "qint16_mixed_float16":
        assert config.parse_from_string('{"precision_config": {"precision_mode": "qint16_mixed_float16"}}').ok()
    elif args.quant_mode == "qint16_mixed_float32":
        assert config.parse_from_string('{"precision_config": {"precision_mode": "qint16_mixed_float32"}}').ok()
    elif args.quant_mode == "force_float32": 
        assert config.parse_from_string('{"precision_config": {"precision_mode": "force_float32"}}').ok()
    elif args.quant_mode == "force_float16": 
        assert config.parse_from_string('{"precision_config": {"precision_mode": "force_float16"}}').ok()

    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
    #mean为0，std为255（即var=65025)。 (input - mean)/std
    assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean": [0, 0, 0], "var": [65025, 65025, 65025]}}}').ok()
    assert config.parse_from_string('{"cross_compile_toolchain_path": "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"}').ok()
    
    if "qint" in args.quant_mode:
        calib_data = FixedCalibData(shape = mm.Dims([args.batch_size, 3, 640, 640]), max_samples = 10, img_dir = args.image_dir)
        calibrator = mm.Calibrator([calib_data])
        if args.remote_addres:
            remote_config = mm.RemoteConfig()
            remote_config.address = args.remote_addres + ":8008"
            calibrator.set_remote(remote_config)

        # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
        assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
        assert calibrator.calibrate(network, config).ok()
        del calibrator
        del calib_data
    
    builder = mm.Builder()
    model = builder.build_model("yolov7_mm_model", network, config)

    assert model != None

    offline_model_name = args.output_model_path
    model.serialize_to_file(offline_model_name)
    print("Generate model done, model save to %s" % offline_model_name) 
