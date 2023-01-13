import numpy as np
import magicmind.python.runtime as mm
import cv2
import os
import glob
import math
from magicmind.python.common.types import get_datatype_by_numpy
from typing import List
class MMCalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, imagelist):
        super().__init__()
        self.imagelist_ = imagelist
        self.shape_ = shape
        self.cur_sample_ = None
        self.cur_data_index_ = 0

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_

    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        dst_h, dst_w = self.shape_.GetDimValue(2), self.shape_.GetDimValue(3)
        for i in range(data_begin, data_end):
            img = cv2.imread(self.imagelist_[i])
            img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
            # mean std
            img = img.astype(np.float32)
            img -= [104, 117, 123]
            # img *= 0.007843
            # HWC to CHW
            img = img.transpose(2, 0, 1)
            imgs.append(np.ascontiguousarray(img)[np.newaxis,:])
        # batch
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0))

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        data_begin = self.cur_data_index_
        data_end = data_begin + batch_size
        if data_end > len(self.imagelist_):
            return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
        self.cur_sample_ = self.preprocess_images(data_begin, data_end)
        self.cur_data_index_ = data_end
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        return mm.Status.OK()

class FixedCalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        print(img_dir)
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.jpg')
        t = glob.glob(img_dir + '/*.JPEG')
        self.data_paths_ += t
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.dst_shape_ = (self.shape_.GetDimValue(2), self.shape_.GetDimValue(3))

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_
    
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        dst_h, dst_w = self.shape_.GetDimValue(2), self.shape_.GetDimValue(3)
        for i in range(data_begin, data_end):
            img = cv2.imread(self.data_paths_[i])
            img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
            # mean std
            img = img.astype(np.float32)
            #img -= 127.5
            img -= [104, 117, 123]
            #img *= 0.007843
            # HWC to CHW
            img = img.transpose(2, 0, 1)
            imgs.append(np.ascontiguousarray(img)[np.newaxis,:])
        # batch
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0))

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        data_begin = self.cur_data_index_
        data_end = data_begin + batch_size
        if data_end > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
        self.cur_sample_ = self.preprocess_images(data_begin, data_end)
        self.cur_data_index_ = data_end
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        return mm.Status.OK()
