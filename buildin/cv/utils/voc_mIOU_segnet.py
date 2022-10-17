import numpy as np
import os 
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.getenv("MAGICMIND_EDGE"), "utils"))
from record_result import write_result

VOC_DATASETS_PATH = os.environ.get("VOC_DATASETS_PATH")
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")

IMAGE_NAME_LIST = str(VOC_DATASETS_PATH) + '/VOC2012/ImageSets/Segmentation/val.txt'
class Evaluator:
    def __init__(self, nclasses):
        self.nclasses_ = nclasses
        self.hist_ = np.zeros((nclasses, nclasses))

    def fast_hist(self, pred, gt):
        '''计算混淆矩阵'''
        # 去掉ground truth中的白色边界（255）
        mask = (gt >= 0) & (gt < self.nclasses_)
        hist = np.bincount(self.nclasses_ * gt[mask].astype(int) + pred[mask], minlength=self.nclasses_ ** 2)
        hist = hist.reshape(self.nclasses_, self.nclasses_)
        return hist

    def evaluate(self, pred, gt):
        self.hist_ += self.fast_hist(pred, gt)

    def miou(self):
        # iou = 对角线上的值(即预测正确的像素数量) / 预测结果和ground truth并集像素数量
        intersection = np.diag(self.hist_)
        union = self.hist_.sum(axis=0) + self.hist_.sum(axis=1) - np.diag(self.hist_)
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.true_divide(intersection, union)
            iou[iou == np.inf] = 0
            iou = np.nan_to_num(iou)
        miou = np.nanmean(iou)
        return miou

with open(IMAGE_NAME_LIST, 'r') as file:
    image_names = file.read().splitlines()

# VOC segmentation 21类（含背景0）
evaluator = Evaluator(21)
for name in tqdm(image_names, "Evaluating"):
    # 载入ground truth
    gt_file = str(VOC_DATASETS_PATH) + '/VOC2012/SegmentationClass/' + name + ".png"
    gt_img = Image.open(gt_file)
    gt = np.array(gt_img).flatten()
    # 载入预测结果
    pred_file = str(PROJ_ROOT_PATH) + '/data/images/' + name + "_result.binary"
    with open(pred_file, 'rb') as file:
        pred = file.read()
        pred = np.frombuffer(pred, dtype=np.uint8)
    # evaluate
    evaluator.evaluate(pred, gt)

print('mIOU : ', evaluator.miou())
write_result(**{"dataset": "voc-mini", "metric":"miou", "eval": evaluator.miou()})
