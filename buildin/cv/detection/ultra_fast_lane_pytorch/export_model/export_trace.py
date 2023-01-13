import torch, os
import sys
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
sys.path.append(str(PROJ_ROOT_PATH) + '/export_model/Ultra-Fast-Lane-Detection/')
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import torch

if __name__ == "__main__":
    args, cfg = merge_config()
    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),
                    use_aux=False).cpu() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = False)

    print(str(PROJ_ROOT_PATH))
    imgs = torch.randn([1, 3, 288, 800], dtype = torch.float)
    net.eval()
    traced_model = torch.jit.trace(net, imgs)
    torch.jit.save(traced_model, str(PROJ_ROOT_PATH)+'/data/models/'+'tusimple_18_traced.pt')
