import sys
import os
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")

sys.path.append( str(PROJ_ROOT_PATH) + '/export_model/Pytorch_Retinaface-b984b4b775b2c4dced95c1eadd195a5c7d32a60b')

import torch
import argparse
from data import cfg_re50
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser()
parser.add_argument("--model_weight", dest = 'model_weight', type=str,default="../data/models/")
parser.add_argument('--input_width', dest = 'input_width', default = 512, type = int, help = 'model input width')
parser.add_argument('--input_height', dest = 'input_height', default = 512, type = int, help = 'model input height')
parser.add_argument('--batch_size', dest = 'batch_size', default = 1, type = int, help = 'model input batch')
parser.add_argument("--traced_pt", dest = 'traced_pt', type=str, default = "../data/models/")


# 载入权重
args = parser.parse_args()
def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
net = RetinaFace(cfg=cfg_re50, phase = 'test')
pretrained_dict = torch.load(args.model_weight, map_location='cpu')
if "state_dict" in pretrained_dict.keys():
    pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
else:
    pretrained_dict = remove_prefix(pretrained_dict, 'module.')
net.load_state_dict(pretrained_dict, strict=False)
net.eval()

# jit.trace
traced_model = torch.jit.trace(net, torch.rand(args.batch_size, 3, args.input_height, args.input_width))
traced_model.save(args.traced_pt)
