import os
import json
import numpy as np
import cv2
import argparse
from PIL import Image 
import magicmind.python.runtime as mm
import magicmind.python.runtime.parser
from calibrator import CalibData, Calibrator
from utils import voc_dataset

PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
VOC_DATASETS_PATH = os.environ.get("VOC_DATASETS_PATH")

parser = argparse.ArgumentParser()
parser.add_argument("--tf_model", type=str, default="../data/models/frozen_inference_graph.pb", help="tf frozen pb model")
parser.add_argument("-i", "--image_dir",  type=str, default=os.path.join(str(os.environ.get("VOC_DATASETS_PATH")), "VOC2012","JPEGImages"), help="VOC2012 datasets")
parser.add_argument("-f", "--file_list", type=str, default=os.path.join(str(os.environ.get("VOC_DATASETS_PATH")), "VOC2012","ImageSets","Segmentation","val.txt"), help="val.txt path")
parser.add_argument("-o", "--output_model_path", type=str, default="../data/models/deeplabv3.mm", help="save mm model to this path")
parser.add_argument("-q", "--quant_mode", type=str, default="qint8_mixed_float16", help="qint8_mixed_float16, qint8_mixed_float32, qint16_mixed_float16, qint16_mixed_float32, forced_float32, forced_float16")
parser.add_argument("-s", "--shape_mutable", type=str, default="false", help="whether the mm model is dynamic or static or not")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("-r", "--remote_addres", type=str, default=os.environ.get('REMOTE_IP', None))


def preprocess(input_image):
    width, height = input_image.size
    resize_ratio = 1.0 * 513 / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    target_size = (513, 513)
    resized_image = input_image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    resized_image = np.asarray(resized_image)
    data = np.copy(resized_image)
    return data

if __name__ == "__main__":
    args = parser.parse_args()

    tf_model = args.tf_model
    file_list = args.file_list
    image_file_path = args.image_dir
    batch_size = args.batch_size
    
    network = mm.Network()
    tf_parser = mm.parser.Parser(mm.ModelKind.kTensorflow)
    tf_parser.set_model_param("tf-model-type", "tf-graphdef-file")
    tf_parser.set_model_param("tf-graphdef-inputs",["ImageTensor:0"])
    tf_parser.set_model_param("tf-graphdef-outputs", ["SemanticPredictions:0"])
    tf_parser.set_model_param("tf-infer-shape", True)
    assert tf_parser.parse(network, tf_model).ok()

    network.get_input(0).set_dimension(mm.Dims([batch_size, 513, 513, 3]))
    
    config = mm.BuilderConfig()
    build_config = {
        "archs": ["mtp_322.10","mtp_372.41"],
        "graph_shape_mutable": False,
        "opt_config": {"type64to32_conversion": True, "conv_scale_fold": True },
        "cross_compile_toolchain_path": "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"
    #    "convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}},
    }
    assert config.parse_from_string(json.dumps(build_config)).ok()


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

    dataset = voc_dataset(file_list = file_list, image_file_path = image_file_path, count = 10)
    sample_data = []
    for data in dataset:
        data = preprocess(data)
        sample_data.append(np.expand_dims(data, 0))

    if "qint" in args.quant_mode:
        calib_data = CalibData(sample_data)
        calibrator = Calibrator(calib_data, mm.QuantizationAlgorithm.LINEAR_ALGORITHM)
        if args.remote_addres:
            remote_config = mm.RemoteConfig()
            remote_config.address = args.remote_addres + ":8008"
            calibrator.set_remote(remote_config)
        assert calibrator.calibrate(network, config).ok()

    builder = mm.Builder()
    model = builder.build_model("deeplabv3_mm_model", network, config)
    assert model != None

    offline_model_name = args.output_model_path
    model.serialize_to_file(offline_model_name)
    print("Generate model done, model save to %s" % offline_model_name) 

