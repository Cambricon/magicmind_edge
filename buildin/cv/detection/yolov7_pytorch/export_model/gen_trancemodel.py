import argparse
import time
import argparse
import torch
 
from models.experimental import attempt_load
from utils.torch_utils import revert_sync_batchnorm, TracedModel
 
 
@torch.no_grad()
def run(opt):
    t = time.time()
 
    imgsz = opt.image_shape
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
 
    weights = opt.weight
 
    device = 'cpu'
 
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # model = revert_sync_batchnorm(model)
    model.eval()
    model.model[-1].ignore_detect_layer=True
    # model.traced = True
 
    rand_example = torch.rand(1, 3, imgsz[0], imgsz[1])
    # model(rand_example)
 
    traced_script_module = torch.jit.trace(model, rand_example, strict=False)
    traced_script_module.save(opt.save_path)
    # Finish
    print("trace_script_model saved to %s" % opt.save_path)
 
 
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", help="pt model path", default="../../data/models/yolov7.pt")
    parser.add_argument("--save_path", help="trace save path", default="../../data/models/yolov7_traced.pt")
    parser.add_argument("--image_shape", help="image shape 640 640", type=int, nargs='+', default=[640, 640])
    opt = parser.parse_args()
    print(opt)
    return opt
 
def main(opt):
    run(opt)
 
 
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)