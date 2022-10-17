# 注意选用合适的ground truth文件，基于imagent2012 devkit训练的模型则使用ILSVRC2012_val.txt,
# 基于imagenet2015 devkit训练的模型则使用ILSVRC2015_val.txt
#GROUND_TRUTH = './output/val_1000.txt'
import argparse
import os
import sys
sys.path.append(os.path.join(os.getenv("MAGICMIND_EDGE"), "utils"))
from record_result import write_result

IMAGENET_DATASETS_PATH = os.environ.get("IMAGENET_DATASETS_PATH")
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
parser = argparse.ArgumentParser()
parser.add_argument( "--ground_truth", type=str, default = os.path.join(str(IMAGENET_DATASETS_PATH), 'val.txt'))
parser.add_argument( "--output", type=str, default= str(PROJ_ROOT_PATH) + '/data/images/')
if __name__ == "__main__":
    args = parser.parse_args()
    GROUND_TRUTH = args.ground_truth
    OUTPUT_DIR = args.output

    top5_error_counts = [0, 0, 0, 0, 0]
    with open(GROUND_TRUTH, 'r') as file:
        lines = file.readlines()
    for line in lines:  # read annotations
        image_name, category_id = line.split()
        category_id = int(category_id)
        output_filename = os.path.splitext(image_name.split('/')[-1])[0] + "_result.txt"
        with open(OUTPUT_DIR  + output_filename, 'r') as file:
            result = file.readline()  # read classification results
            top5 = result.split()
            assert len(top5) == 5
            try:
                error_end_pos = top5.index(str(category_id))
            except ValueError:
                error_end_pos = 5
            for i in range(error_end_pos):
                top5_error_counts[i] = top5_error_counts[i] + 1
    
    total_count = len(lines)
    for i in range(5):
        print('top %d accuracy: %f' % (i + 1, 1 - top5_error_counts[i] / total_count))
    # res = {}
    # for i in range(5):
    #     res["top%d"%(i+1)] = "%.3f"%(1 - top5_error_counts[i] / total_count)
    top5 = 1 - top5_error_counts[4] / total_count
    write_result(**{"dataset": "imagenet-mini", "metric":"top5", "eval": top5})

