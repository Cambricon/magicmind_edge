import numpy as np
from PIL import Image
import os 

PROJ_ROOT_PATH = os.environ.get('PROJ_ROOT_PATH')
VOC_DATASETS_PATH = os.environ.get('VOC_DATASETS_PATH')

IMAGE_DIR = str(VOC_DATASETS_PATH) + '/VOC2012/JPEGImages/'
# 选取一张图片，并查看其图像分割结果
TEST_IMAGE_NAME = '2007_000464'

# 载入原图
image_file = IMAGE_DIR + '/' + TEST_IMAGE_NAME + '.jpg'
img = Image.open(image_file)
# 载入ground truth
gt_file = str(VOC_DATASETS_PATH) + '/VOC2012/SegmentationClass/' + TEST_IMAGE_NAME + ".png"
gt_img = Image.open(gt_file)
# 载入预测结果
result_file = str(PROJ_ROOT_PATH) + '/data/images/' + TEST_IMAGE_NAME + '_result.binary'
with open(result_file, 'rb') as file:
    result = file.read()
result = np.frombuffer(result, dtype=np.uint8)
result = result.reshape(img.size[1], img.size[0])
result = Image.fromarray(result, mode='P')
# 将ground truth中使用的调色板应用到预测结果上
result.putpalette(gt_img.getpalette())

# 显示预测图
print('result png save in current path.')
#display(result)
result.save(TEST_IMAGE_NAME + '_result.png')
