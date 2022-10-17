import os
import shutil
import random

old_root_dir = os.path.join("/zoo_ci", "datasets", "pascal_voc_seg", "VOC2012") 
new_root_dir = os.path.join("/zoo_ci", "datasets-mini", "VOCdevkit", "VOC2012")

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if os.path.exists(new_root_dir):
    shutil.rmtree(new_root_dir)

jpeg_dir = os.path.join(new_root_dir, "JPEGImages")
annotations_dir = os.path.join(new_root_dir, "Annotations")
imageset_dir = os.path.join(new_root_dir, "ImageSets")
segmentationClass_dir = os.path.join(new_root_dir, "SegmentationClass")
segmentationObject_dir = os.path.join(new_root_dir, "SegmentationObject")

imageset_action_dir = os.path.join(imageset_dir, "Action")
imageset_layout_dir = os.path.join(imageset_dir, "Layout")
imageset_main_dir = os.path.join(imageset_dir, "Main")
imageset_segmentation_dir = os.path.join(imageset_dir, "Segmentation")

for dir in [new_root_dir, 
            jpeg_dir, 
            annotations_dir,
            imageset_dir, 
            segmentationClass_dir, 
            segmentationObject_dir,
            imageset_action_dir,
            imageset_layout_dir,
            imageset_main_dir,
            imageset_segmentation_dir
            ]:
    create_dir(dir)

seg_val_file = os.path.join(old_root_dir, "ImageSets", "Segmentation", "val.txt")
main_val_file = os.path.join(old_root_dir, "ImageSets", "Main", "val.txt")

with open(seg_val_file, "r") as f:
    seg_lines = f.readlines()
    random.shuffle(seg_lines)

with open(main_val_file, "r") as f:
    main_lines = f.readlines()
    random.shuffle(main_lines)

inter_set = set(main_lines).intersection(set(seg_lines))

f_seg_w = open(os.path.join(imageset_segmentation_dir, "val.txt"), "w")

for line in inter_set:
    line = line.strip()
    jpg_file = os.path.join(old_root_dir,"JPEGImages","%s.jpg"%line)
    seg_class_file = os.path.join(old_root_dir,"SegmentationClass","%s.png"%line)
    seg_obj_file = os.path.join(old_root_dir,"SegmentationObject","%s.png"%line)
    for file in [jpg_file, seg_class_file, seg_obj_file]:
        if not os.path.exists(file):
            print("file:%s not exist"%file)
            exit(1)
    
    shutil.copy(jpg_file, jpeg_dir)
    shutil.copy(seg_class_file, segmentationClass_dir)
    shutil.copy(seg_obj_file, segmentationObject_dir)
    f_seg_w.write(line + "\n")

f_seg_w.close()

# shutil.copy(seg_val_file, imageset_segmentation_dir)


    
f_main_w = open(os.path.join(imageset_main_dir, "val.txt"), "w")
for line in inter_set:
    line = line.strip()
    jpg_file = os.path.join(old_root_dir,"JPEGImages","%s.jpg"%line)
    annotations_file = os.path.join(old_root_dir,"Annotations","%s.xml"%line)
   
    for file in [jpg_file, annotations_file]:
        if not os.path.exists(file):
            print("file:%s not exist"%file)
            exit(1)
    
    shutil.copy(jpg_file, jpeg_dir)
    shutil.copy(annotations_file, annotations_dir)
    f_main_w.write(line + "\n")
f_main_w.close()

# shutil.copy(main_val_file, imageset_main_dir)
