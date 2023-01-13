import os, sys, json, scipy
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
sys.path.append(str(PROJ_ROOT_PATH) + '/export_model/Ultra-Fast-Lane-Detection/')
sys.path.append(os.path.join(os.getenv("MAGICMIND_EDGE"), "utils"))
from record_result import write_result
from utils.common import merge_config
from utils.dist_utils import is_main_process, dist_print, get_rank, get_world_size, dist_tqdm, synchronize
from evaluation.tusimple.lane import LaneEval
from utils.dist_utils import dist_print 

def combine_tusimple_test(work_dir,exp_name):
    size = get_world_size()
    all_res = []
    for i in range(size):
        output_path = os.path.join(work_dir,exp_name+'.%d.txt'% i)
        with open(output_path, 'r') as fp:
            res = fp.readlines()
        all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        pos = res.find('clips')
        name = res[pos:].split('\"')[0]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(work_dir,exp_name+'.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(all_res_no_dup)
        
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

    # eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir, cfg.griding_num, False, distributed)
    exp_name = 'tusimple_eval_tmp'
    combine_tusimple_test(cfg.test_work_dir,exp_name)
    res = LaneEval.bench_one_submit(os.path.join(cfg.test_work_dir,exp_name + '.txt'),os.path.join(cfg.data_root,'test_label.json'))
    res = json.loads(res)
    for r in res:
        dist_print(r['name'], r['value'])
    write_result(**{"dataset": "Tusimple", "metric":"Acc", "eval": res[0]['value']})
