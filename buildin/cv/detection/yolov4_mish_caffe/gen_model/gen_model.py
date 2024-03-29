import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import numpy as np
import glob
import os 
import argparse
import cv2
from calibrator import CalibData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "yolv4-mish caffe model calibrartion and build")
    parser.add_argument('--quant_mode', type=str,   default='force_float16', required=True ,help='Quant_mode')
    parser.add_argument('--batch_size', type=int,   default=8, required=True ,help='batch_size')
    parser.add_argument('--datasets_dir', type=str, default="false", required=True ,help='datasets_dir')
    parser.add_argument('--caffe_prototxt', type=str, default="false", required=True ,help='caffe_prototxt')
    parser.add_argument('--caffe_model', type=str, default="false", required=True ,help='caffe_model')
    parser.add_argument('--mm_model', type=str, default="", required=True ,help='mm model save path')
    parser.add_argument("--remote_addres", type=str, default=os.environ.get('REMOTE_IP', None))
    args = parser.parse_args()

    DEV_ID = 0
    PAD_VALUE = 128
    BATCH_SIZE = args.batch_size
    INPUT_SIZE = (416, 416) # h x w
    IMAGE_DIR = args.datasets_dir #'val2017'
    PROTOTXT = args.caffe_prototxt
    CAFFEMODEL = args.caffe_model
    ANCHORS = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
    MM_MODEL = args.mm_model
    CALIB_SAMPLES_DIR = IMAGE_DIR
    MAX_CALIB_SAMPLES = 10

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
    assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean": [0, 0, 0], "var": [65025, 65025, 65025]}}}').ok()
    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()

    # 打开设备

    if args.quant_mode != "force_float16" and args.quant_mode != "force_float32":
        print("Calibraing...")
        # 创建量化工具并设置量化统计算法
        calib_data = CalibData(mm.Dims((BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1])), MAX_CALIB_SAMPLES, CALIB_SAMPLES_DIR,PAD_VALUE)
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

    perms = [0, 2, 3, 1]  # 0 : N, 1 : C, 2 : H, 3 : W
    const_node = network.add_i_const_node(mm.DataType.INT32, mm.Dims(
        [len(perms)]), np.array(perms, dtype=np.int32))
    output_tensors = []
    for i in range(network.get_output_count()):
        # 添加premute算子做NCHW到NHWC的转换
        tensor = network.get_output(i)
        permute_node = network.add_i_permute_node(
            tensor, const_node.get_output(0))
        output_tensors.append(permute_node.get_output(0))
    output_count = network.get_output_count()
    for i in range(output_count):
        # 去掉原网络输出tensor标志
        network.unmark_output(network.get_output(0))

    # anchors
    bias_buffer = ANCHORS
    bias_node = network.add_i_const_node(mm.DataType.FLOAT32, mm.Dims([len(bias_buffer)]),
        np.array(bias_buffer, dtype=np.float32))
    # yolov4后处理算子
    detect_out = network.add_i_detection_output_node(
        output_tensors, bias_node.get_output(0))
    detect_out.set_layout(mm.Layout.NONE, mm.Layout.NONE)
    detect_out.set_algo(mm.IDetectionOutputAlgo.YOLOV4)
    detect_out.set_nms_type(mm.INmsType.DIOU_NMS)
    detect_out.set_confidence_thresh(0.001)
    detect_out.set_nms_thresh(0.45)
    detect_out.set_diou_power_exp(0.6)
    detect_out.set_aspect_ratios([1.2, 1.1, 1.05])
    detect_out.set_scale(1.0)
    detect_out.set_num_coord(4)
    detect_out.set_num_class(80)
    detect_out.set_num_entry(5)
    detect_out.set_num_anchor(3)
    detect_out.set_num_box_limit(256)
    detect_out.set_image_shape(INPUT_SIZE[0], INPUT_SIZE[1])

    # 将detect_out层输出标记为网络输出
    detection_output_count = detect_out.get_output_count()
    for i in range(detection_output_count):
        network.mark_output(detect_out.get_output(i))

    # 生成模型
    builder = mm.Builder()
    assert builder is not None
    mm_model = builder.build_model("magicmind model", network, config)
    assert mm_model is not None
    # 将模型序列化为离线文件
    assert mm_model.serialize_to_file(MM_MODEL).ok()
    # 由于使用了insert_bn_before_firstnode参数来构建模型，可以看到模型输入是UINT8类型
    #mm_model.get_input_data_type(0)
