import sys
sys.path.append('crnn-pytorch/src')
from model import CRNN
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_weight", dest = 'model_weight', type=str,default="../data/models/crnn.pth")
parser.add_argument('--input_width', dest = 'input_width', default = 100, type = int, help = 'model input width')
parser.add_argument('--input_height', dest = 'input_height', default = 32, type = int, help = 'model input height')
parser.add_argument('--batch_size', dest = 'batch_size', default = 1, type = int, help = 'model input batch')
parser.add_argument("--traced_pt", dest = 'traced_pt', type=str, default = "../data/models/")

args = parser.parse_args()
# 载入权重
model = CRNN(1, args.input_height, args.input_width, 37,
                map_to_seq_hidden=64,
                rnn_hidden=256,
                leaky_relu=False)
model.load_state_dict(torch.load(args.model_weight,map_location='cpu'))
model.eval()

# jit.trace
traced_model = torch.jit.trace(model, torch.rand(args.batch_size, 1, args.input_height, args.input_width))
traced_model.save(args.traced_pt)
