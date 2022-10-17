import os
PROJ_ROOT_PATH = os.environ.get('PROJ_ROOT_PATH')
VOC_DATASETS_PATH = os.environ.get('VOC_DATASETS_PATH')
# 读取测试图片文件名
filenames = []
with open(str(VOC_DATASETS_PATH) + '/VOC2012/ImageSets/Main/val.txt', 'r') as f:
    filenames = f.readlines()

classes = ('__background__',
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
voc_preds_files = []

for t in classes:
    voc_preds_files.append(open( str(PROJ_ROOT_PATH) + '/data/voc_pred/comp3_det_val_' + t + '.txt', 'w'))

for filename in filenames:
    detections_filepath = str(PROJ_ROOT_PATH) +  '/data/images/' + filename.strip() + '.txt'
    with open(detections_filepath, 'r') as f:
        detections = f.readlines()
    for bbox_str in detections:
        # category, score, left, top, right, bottom
        bbox = bbox_str.split()
        # write filename first
        voc_preds_files[int(bbox[0])].write(filename.strip())
        for i in range(1, 6):
            # write bbox
            voc_preds_files[int(bbox[0])].write(' ' + bbox[i])
        voc_preds_files[int(bbox[0])].write('\n')
for file in voc_preds_files:
    file.close()
