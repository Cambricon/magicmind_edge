import os
import json
import random
import shutil

with open("/zoo_ci/datasets/coco/annotations/instances_val2017.json", "r") as f:
    ori_coco = json.loads(f.read())

random.shuffle(ori_coco["images"])

new_coco = dict()
new_coco["info"] = ori_coco["info"]
new_coco["licenses"] = ori_coco["licenses"]
new_coco["categories"] = ori_coco["categories"]

new_coco["images"] = ori_coco["images"][:1000]
new_coco["annotations"] = list()

image_ids = [x["id"] for x in new_coco["images"]]
image_files = [x["file_name"] for x in new_coco["images"]]

for annotation in ori_coco["annotations"]:
    if annotation["image_id"] in image_ids:
        new_coco["annotations"].append(annotation)

if not os.path.exists("/zoo_ci/datasets-mini/coco_new/annotations"):
    os.makedirs("/zoo_ci/datasets-mini/coco_new/annotations")

if not os.path.exists("/zoo_ci/datasets-mini/coco_new/val2017"):
    os.makedirs("/zoo_ci/datasets-mini/coco_new/val2017")

json_str = json.dumps(new_coco,indent=4)
with open("/zoo_ci/datasets-mini/coco_new/annotations/instances_val2017.json", 'w') as json_file:
    json_file.write(json_str)

for file in image_files:
    src_path = os.path.join("/zoo_ci/datasets/coco/val2017", file)
    dst_path = os.path.join("/zoo_ci/datasets-mini/coco_new/val2017", file)
    shutil.copy(src_path, dst_path)




# info.keys()
# 'info', 'licenses', 'images', 'annotations', 'categories'

# print(ori_coco.keys())

# print(info['info'])
# image_ids = [x["id"] for x in ori_coco["images"]]
# annotations_ids = [x["image_id"] for x in ori_coco["annotations"]]
# categories_ids =[x["id"] for x in info["categories"]]


# print(ori_coco["annotations"][0])


# print(len(image_ids), len(set(annotations_ids)))
# print(info["images"][0]["id"])


# print(info["annotations"][0]["id"])

# print(info["categories"][0]["id"])
