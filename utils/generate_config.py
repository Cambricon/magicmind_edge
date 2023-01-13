import os
from types import new_class
import yaml

branch_name = os.environ.get("CI_COMMIT_REF_NAME")
commit_message = os.environ.get("CI_COMMIT_MESSAGE", "")

template_file = "utils/data/template.yaml"
search_dirs = [
    "buildin/cv/classification", 
    "buildin/cv/detection", 
    "buildin/cv/segmentation",
    "buildin/cv/ocr",
    "buildin/cv/other"]

black_list = [
    "buildin/cv/classification/vgg16_caffe"
]

CI_SOURCE = os.getenv("CI_PIPELINE_SOURCE", None)

# if CI_SOURCE == "push":
dirs = []
for search_dir in search_dirs:
    s_dir = os.listdir(search_dir)
    for d in s_dir:
        dir = os.path.join(search_dir, d)
        if os.path.isdir(dir):
            dirs.append(dir)

with open(template_file, "r") as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        
if branch_name != "main" and "all_case" not in commit_message:
    new_dir = []
    for dir in dirs:
        name = os.path.split(dir)[-1]
        if name in commit_message:
            new_dir.append(dir)
    dirs = new_dir

for bl in black_list:
    if bl in dirs:
        dirs.remove(bl)

for dir in dirs:
    project_name = os.path.split(dir)[-1]
    cfg[f"{project_name}"] = {
        "extends": ".network_test",
        "variables": {
        "TEST_PROJ_DIR": f"{dir}"
        } 
    }

with open("jobs.yml", "w", encoding="utf-8") as f:
    yaml.dump(data=cfg, stream=f, allow_unicode=True)

project_names = [os.path.split(x)[-1] for x in dirs]
project_names.sort()
print(f"Generate jobs done, {len(dirs)} jobs ")
print("Job list: ")
for name in project_names:
    print(f"    {name}")
# else:
#     with open("utils/data/ide.yaml", "r") as f:
#         cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
#     with open("jobs.yml", "w", encoding="utf-8") as f:
#         yaml.dump(data=cfg, stream=f, allow_unicode=True)
