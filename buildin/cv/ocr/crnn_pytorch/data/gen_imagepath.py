# 生成输入图片列表文件
import glob
import os
MJ_DATASETS_PATH = os.environ.get("MJ_DATASETS_PATH")
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
files=glob.glob(str(MJ_DATASETS_PATH) + '/mjsynth_mini/3000/*/*.jpg')
with open(str(PROJ_ROOT_PATH) + '/data/image_list.txt', 'w') as f:
    for t in files:
        f.write(t)
        f.write('\n')
print("You should make sure your MJ_DATASETS_PATH and 3226_DATASETS_PATH are the same ")