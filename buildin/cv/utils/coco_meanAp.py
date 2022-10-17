from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob
import os
import sys
sys.path.append(os.path.join(os.getenv("MAGICMIND_EDGE"), "utils"))
from record_result import write_result

COCO_DATASETS_PATH = os.environ.get("COCO_DATASETS_PATH")
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
DETECT_RESULT_FILE = os.path.join(str(PROJ_ROOT_PATH), "data/result.json")
coco_gt = COCO(os.path.join(str(COCO_DATASETS_PATH), 'annotations/instances_val2017.json'))
print(DETECT_RESULT_FILE)
coco_dt = coco_gt.loadRes(DETECT_RESULT_FILE)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.params.imgIds = coco_gt.getImgIds()
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
write_result(**{"dataset": "coco-mini", "metric":"map", "eval": coco_eval.stats[1]})
