import magicmind.python.runtime as mm
import numpy as np
import glob
import os
import cv2
import math

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, img_dir: str, calibrate_list: str):
        super().__init__()
        self.shape_ = shape
        with open(calibrate_list, 'r') as f:
                image_paths = f.readlines()
        self.images = []
        for image_path in image_paths:
            image = cv2.imread(img_dir + '/' + image_path.strip())
            assert image is not None, 'image [' + image_path.strip() + '] not exists!'
            self.images.append(image)
        nimages = len(self.images)
        assert nimages != 0, 'no images in calibrate list[' + calibrate_list + ']!'
        # at least one batch
        if nimages < self.shape_.GetDimValue(0):
            for i in range(self.shape_.GetDimValue(0) - nimages):
                self.images.append(self.images[0])
        self.cur_image_index_ = 0

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_

    def preprocess_image(self):
        if self.cur_image_index_ == len(self.images):
            return None
        h = self.shape_.GetDimValue(2)
        w = self.shape_.GetDimValue(3)
        image = self.images[self.cur_image_index_]
        scaling_factor = min((h - 1) / (image.shape[0] - 1), (w - 1) / (image.shape[1] - 1))
        m = np.zeros((2, 3))
        m.astype('float64')
        m[0, 0] = scaling_factor
        m[1, 1] = scaling_factor
        image = cv2.warpAffine(image, m, (w, h),
                flags = cv2.INTER_CUBIC if scaling_factor > 1.0 else cv2.INTER_AREA,
                borderMode = cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        image = image.astype('float32')
        image -= 128
        image /= 256.0
        image = np.transpose(image, (2, 0, 1)) # HWC >>> CHW
        self.cur_image_index_ = self.cur_image_index_ + 1
        return image

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        preprocessed_images = []
        for i in range(batch_size):
            image = self.preprocess_image()
            if image is None:
                # no more data
                return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
            preprocessed_images.append(image)
        self.cur_sample_ = np.array(preprocessed_images)
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_image_index_ = 0
        return mm.Status.OK()