import os
import json
import numpy as np
import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType
from calibrator import FixedCalibData, MMCalibData

PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
VOC_DATASETS_PATH = os.environ.get("VOC_DATASETS_PATH")

parser = argparse.ArgumentParser()
parser.add_argument( "--prototxt", type=str, default= str(PROJ_ROOT_PATH) + '/data/models/segnet_pascal.prototxt', help="prototxt file")
parser.add_argument( "--caffe_model", type=str, default= str(PROJ_ROOT_PATH) + '/data/models/segnet_pascal.caffemodel', help="caffemodel file")
parser.add_argument( "--image_dir",  type=str, default= str(VOC_DATASETS_PATH) + '/VOC2012/JPEGImages/', help="imagenet datasets")
parser.add_argument( "--output_model_path", type=str, default= str(PROJ_ROOT_PATH) + '../data/models/segnet_pascal_qint8_mixed_float16_1.mm', help="save mm model to this path")
parser.add_argument( "--quant_mode", type=str, default="qint8_mixed_float16", help="qint8_mixed_float16, qint8_mixed_float32, qint16_mixed_float16, qint16_mixed_float32, forced_float32, forced_float16")
parser.add_argument( "--batch_size", type=int, default=1, help="batch_size")
parser.add_argument( "--remote_addres", type=str, default=os.environ.get('REMOTE_IP', None))

if __name__ == "__main__":
    args = parser.parse_args()
    
    network = mm.Network()
    caffe_parser = Parser(mm.ModelKind.kCaffe)
    assert caffe_parser.parse(network, args.caffe_model, args.prototxt).ok()
    input_size = [224, 224] 
    MEAN = [0, 0, 0]
    STD = [1.0, 1.0, 1.0]
    network.get_input(0).set_dimension(mm.Dims((args.batch_size, 3, input_size[0], input_size[1])))
    
    config = mm.BuilderConfig()
    assert config.parse_from_string('{"archs":["mtp_322.10"]}')
    # assert config.parse_from_string('{"archs":["mtp_372.41"]}')
    #assert config.parse_from_string('{"archs":["mtp_322.10", "mtp_372"]}')
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}')
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}')
    assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}')
    # 本例中提供的图像分割程序按NHWC的方式读取网络输出，故此处将输出转为NHWC摆放顺序
    assert config.parse_from_string('{"convert_output_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    
    ###static or dynamic model
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
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_axis"}}').ok()
    var = list(map(lambda num:num*num, STD))
    assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean":' + str(MEAN) + \
                                ', "var": ' + str(var) + '}}}').ok()
    assert config.parse_from_string('{"cross_compile_toolchain_path": "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"}').ok()
    if "qint" in args.quant_mode:
        calib_data = FixedCalibData(shape = mm.Dims([args.batch_size, 3, input_size[0], input_size[1]]), max_samples = 10, img_dir = args.image_dir)
        calibrator = mm.Calibrator([calib_data])
        if args.remote_addres:
            remote_config = mm.RemoteConfig()
            remote_config.address = args.remote_addres + ":8008"
            calibrator.set_remote(remote_config)

        # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
        assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
        assert calibrator.calibrate(network, config).ok()
    # 当使用了insert_bn_before_firstnode参数后，不需要设置网络输入数据类型，且生成的网络的输入数据类型为UINT8。
    builder = mm.Builder()
    model = builder.build_model("segnet_caffe_model", network, config)

    assert model != None

    offline_model_name = args.output_model_path
    model.serialize_to_file(offline_model_name)
    print("Generate model done, model save to %s" % offline_model_name) 
