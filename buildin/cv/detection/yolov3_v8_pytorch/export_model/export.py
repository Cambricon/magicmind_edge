from models import Darknet
import torch
import os
import argparse

PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("--cfg", type=str, default='cfg/yolov3.cfg', help="cfg file")
parser.add_argument('--weights', type=str, default= str(PROJ_ROOT_PATH) + '/data/models/yolov3.pt', help='weights path')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416, 416], help='image (h, w)')
parser.add_argument("--traced_pt", type=str, default= str(PROJ_ROOT_PATH) + '/data/models/yolov3_traced.pt', help="traced pt file")

class Yolov3(Darknet):
    def __init__(self, cfg, img_size=(416, 416)):
        super(Yolov3, self).__init__(cfg, img_size)

    def forward_once(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:  # sum, concat
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                # 跳过YOLOLayer
                yolo_out.append(x)
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
        return tuple(yolo_out)
if __name__ == "__main__":
    args = parser.parse_args()
    model = Yolov3(args.cfg, args.imgsz)
    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model'])
    model.eval()
    model.fuse()
    traced_model = torch.jit.trace(model, torch.rand(args.batch_size, 3, args.imgsz[0], args.imgsz[1]), check_trace=False)
    torch.jit.save(traced_model, args.traced_pt)
