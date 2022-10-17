import os
from re import S
from PIL import Image
import cv2
from cv2 import resize
import numpy as np
import logging
import argparse
import sys
sys.path.append(os.path.join(os.getenv("MAGICMIND_EDGE"), "utils"))
from record_result import write_result

VOC_DATASETS_PATH = os.environ.get("VOC_DATASETS_PATH")
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")

def voc_dataset(file_list, image_file_path, count):
    with open(file_list, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    current_count = 0
    for line in lines:
        image_name = line.replace("\n", "")
        image_path = os.path.join(image_file_path, image_name + ".jpg")
        img = Image.open(image_path)
        yield img
        current_count += 1
        if current_count > count and count != -1:
            break
class Record:
    def __init__(self, filename):
        self.file = open(os.path.join(filename), "w")

    def write(self, line, _print = False):
        self.file.write(line + "\n")
        if _print:
            print(line)

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
 
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type = str, default = str(VOC_DATASETS_PATH) + '/VOC2012/SegmentationClass', help ='directory which stores VOC Segmentation val gt images')
    parser.add_argument('--pred_dir', type = str, default = str(PROJ_ROOT_PATH) + '/data/images/', help ='directory which stores VOC Segmentation val pred images')
    parser.add_argument('--file_list', type = str, default = str(VOC_DATASETS_PATH) + '/VOC2012/ImageSets/Segmentation/val.txt', help='file list which uses to compute mIOU')
    args = parser.parse_args()
    image_path = args.pred_dir
    gt_path = args.gt_dir
    val_txt = args.file_list

    num_classes = 21
    name_classes = ["backgroud", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    hist = np.zeros((num_classes, num_classes))
    
    with open(val_txt, "r") as f:
        lines = f.readlines()
    #record = Record("eval_result.txt")
    for line in lines:
        line = line.strip()
        image_file = os.path.join(image_path, line + ".png")

        # load gt
        gt_file_path = os.path.join(gt_path, line + ".png")
        label_org = Image.open(gt_file_path)
        print(image_file)
        pred_image = Image.open(image_file)
        pred_image = pred_image.resize(label_org.size)
        pred_p = pred_image.quantize(palette = label_org) # convert to P mode

        label = np.array(label_org)
        pred = np.array(pred_p)
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)), True)
        #record.write('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)), True)
    #record.write('mIOU:' + str(round(np.nanmean(mIoUs) * 100, 2)), True)
    print('mIOU:' + str(round(np.nanmean(mIoUs) * 100, 2)))
    write_result(**{"dataset": "voc-mini", "metric":"miou", "eval": np.nanmean(mIoUs)})
