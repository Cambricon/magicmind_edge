import os
import shutil
import random
from typing import Counter

COUNT = 1300

old_root_dir = os.path.join("/zoo_ci", "datasets", "IJB_release", "IJBC") 
new_root_dir = os.path.join("/zoo_ci", "datasets-mini", "IJB_release", "IJBC")

def read_txt(file):
    with open(file, "r") as f:
        lines = f.readlines()
    return lines

def write_txt(file, lines):
    with open(file, "w") as f:
        lines = f.writelines(lines)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if os.path.exists(new_root_dir):
    shutil.rmtree(new_root_dir)
    
create_dir(new_root_dir)
create_dir(os.path.join(new_root_dir, "meta"))
create_dir(os.path.join(new_root_dir, "loose_crop"))

old_face_tid_mid_path = os.path.join(old_root_dir, "meta", "ijbc_face_tid_mid.txt")
new_face_tid_mid_path = os.path.join(new_root_dir, "meta", "ijbc_face_tid_mid.txt")

old_name_5pts_score_path = os.path.join(old_root_dir, "meta", "ijbc_name_5pts_score.txt")
new_name_5pts_score_path = os.path.join(new_root_dir, "meta", "ijbc_name_5pts_score.txt")

old_template_pair_label_path = os.path.join(old_root_dir, "meta", "ijbc_template_pair_label.txt")
new_template_pair_label_path = os.path.join(new_root_dir, "meta", "ijbc_template_pair_label.txt")

old_image_dir = os.path.join(old_root_dir, "loose_crop")
new_image_dir = os.path.join(new_root_dir, "loose_crop")

old_template_pair_label = read_txt(old_template_pair_label_path)
new_template_pair_label = []

pos_list = []
neg_list = []

old_template_pair_label

for line in old_template_pair_label:
    t1, t2, label = line.split(" ")
    if int(label) == 0:
        neg_list.append(line)
    else:
        pos_list.append(line)

ratio = len(neg_list) / len(pos_list)

neg_count = round(COUNT * (len(neg_list)/(len(neg_list) + len(pos_list))))
pos_count = round(COUNT * (len(pos_list)/(len(neg_list) + len(pos_list))))

print("pos count", pos_count)
print("neg count", neg_count)

random.shuffle(neg_list)
random.shuffle(pos_list)

neg_objs = random.choices(neg_list, k=neg_count)
pos_objs = random.choices(pos_list, k=pos_count)

new_template_pair_label = neg_objs + pos_objs

write_txt(new_template_pair_label_path, new_template_pair_label)

tids = set()

for line in new_template_pair_label:
    t1, t2, label = line.split(" ")
    tids.add(t1)
    tids.add(t2)
    
old_face_tid_mid = read_txt(old_face_tid_mid_path)
new_face_tid_mid = [] 

id_to_info = {}

for line in old_face_tid_mid:
    img, tid, mid = line.split(" ")
    if tid not in tids:
        continue
    if tid not in id_to_info.keys():
        id_to_info[tid] = [line]
    else:
        id_to_info[tid].append(line)

imgs = set()
   
for k in id_to_info.keys():
    info = id_to_info[k]
    max = 3
    if len(info) < 3:
        max = len(info)
    objs = random.choices(info, k= random.randint(1, max))
    for obj in objs:
        new_face_tid_mid.append(obj)
        img, tid, mid = obj.split(" ")
        imgs.add(img)

write_txt(new_face_tid_mid_path, new_face_tid_mid)
        
old_name_5pts_score = read_txt(old_name_5pts_score_path)

new_name_5pts_score = []

for line in old_name_5pts_score:
    img = line.split(" ")[0]
    if img in imgs:
        new_name_5pts_score.append(line)
    
write_txt(new_name_5pts_score_path, new_name_5pts_score)


for img in imgs:
    shutil.copy(os.path.join(old_image_dir, img), os.path.join(new_image_dir, img))