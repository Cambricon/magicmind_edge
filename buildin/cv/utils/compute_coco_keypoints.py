import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import sys
sys.path.append(os.path.join(os.getenv("MAGICMIND_EDGE"), "utils"))
from record_result import write_result

def main():
    args = argparse.ArgumentParser(description='Evaluate')
    args.add_argument('--ann_file', dest = 'ann_file', required = True, type = str, help = 'annotation file path')
    args.add_argument('--res_file', dest = 'res_file', required = True, type = str, help = 'result file path')
    args.add_argument('--res2_file', dest = 'res2_file', required = False, type = str, help = 'result file path')
    args = args.parse_args()

    coco_gt = COCO(args.ann_file)
    coco_dt = coco_gt.loadRes(args.res_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.params.imgIds = coco_gt.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stat1 = coco_eval.stats[1]
    stat2 = 0.0;
    if args.res2_file is not None:
        coco_gt = COCO(args.ann_file)
        coco_dt = coco_gt.loadRes(args.res2_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
        coco_eval.params.imgIds = coco_gt.getImgIds()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stat2 = coco_eval.stats[1]
        
    write_result(**{"dataset": "coco", "metric":"AP", "eval": coco_eval.stats[1]})

if __name__ == "__main__":
    main()

