# 生成输入图片列表文件
import glob
import os
WIDER_VAL_PATH = os.environ.get("WIDER_VAL_PATH")
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
files=glob.glob(str(WIDER_VAL_PATH) + '/images/*/*.jpg')
with open(str(PROJ_ROOT_PATH) + '/data/image_list.txt', 'w') as f:
    for t in files:
        f.write(t)
        f.write('\n')