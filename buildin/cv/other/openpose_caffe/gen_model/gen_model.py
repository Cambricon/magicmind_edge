import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import numpy as np
import glob
import os 
import argparse
import cv2
from calibrator import CalibData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "openpose caffe model calibrartion and build")
    parser.add_argument('--quant_mode', type=str,   default='force_float16', required=True ,help='Quant_mode')
    parser.add_argument('--batch_size', type=int,   default=8, required=True ,help='batch_size')
    parser.add_argument('--datasets_dir', type=str, default="false", required=True ,help='datasets_dir')
    parser.add_argument('--caffe_prototxt', type=str, default="false", required=True ,help='caffe_prototxt')
    parser.add_argument('--caffe_model', type=str, default="false", required=True ,help='caffe_model')
    parser.add_argument('--mm_model', type=str, default="", required=True ,help='caffe_model')
    parser.add_argument("--remote_addres", type=str, default=os.environ.get('REMOTE_IP', None))
    parser.add_argument('--calibrate_list', type=str, default = 'calibrate_list.txt', help = 'image list file path, file contains input image paths for calibration')
    args = parser.parse_args()

    DEV_ID = 0
    PAD_VALUE = 128
    BATCH_SIZE = args.batch_size
    INPUT_SIZE = (368, 656) # h x w
    IMAGE_DIR = args.datasets_dir #'val2017'
    PROTOTXT = args.caffe_prototxt
    CAFFEMODEL = args.caffe_model
    MM_MODEL = args.mm_model
    CALIB_SAMPLES = args.calibrate_list
    
    parser = Parser(mm.ModelKind.kCaffe)
    network = mm.Network()
    assert parser.parse(network, CAFFEMODEL, PROTOTXT).ok()
    assert network.get_input(0).set_dimension(mm.Dims((BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]))).ok()

    config = mm.BuilderConfig()
    precision_json_str = '{"precision_config" : { "precision_mode" : "%s" }}'%args.quant_mode
    assert config.parse_from_string(precision_json_str).ok()

    assert config.parse_from_string("{\"opt_config\":{\"type64to32_conversion\":true}}").ok()
    assert config.parse_from_string("{\"opt_config\":{\"conv_scale_fold\":true}}").ok()
    # 禁用模型输入输出规模可变功能
    assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    assert config.parse_from_string("""{"cross_compile_toolchain_path": "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"}""").ok()
    # 硬件平台
    assert config.parse_from_string("""{"archs": ["mtp_322"]}""").ok()
    # 将网络输入数据摆放顺序由NCHW转为NHWC
    assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    # 将预处理中标准化过程集成到模型中(img = (img - mean) / std), 其中var的值为std的平方
    assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean": [128,128,128], "var": [65536,65536,65536]}}}').ok()
    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
    
    # 打开设备
    
    print("Calibraing...")
    # 创建量化工具并设置量化统计算法
    calib_data = CalibData(mm.Dims((BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1])), IMAGE_DIR, CALIB_SAMPLES)
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None
    if args.remote_addres:
        remote_config = mm.RemoteConfig()
        remote_config.address = args.remote_addres + ":8008"
        calibrator.set_remote(remote_config)
        
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    assert calibrator.calibrate(network, config).ok()
    del calib_data
    del calibrator
    print("Calibra Done!")
    builder = mm.Builder()
    assert builder is not None
    mm_model = builder.build_model("magicmind model", network, config)
    assert mm_model is not None
    # 将模型序列化为离线文件
    assert mm_model.serialize_to_file(MM_MODEL).ok()
