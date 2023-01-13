import os
import sys
sys.path.append(os.path.join(os.getenv("MAGICMIND_EDGE"), "utils"))
from record_result import write_result
file_path = os.environ.get('PROJ_ROOT_PATH')
outputlog = file_path +"/export_model/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet18/L1BalanceCELoss/output.log"
with open(outputlog, 'r') as f:
    lines = f.readlines()
    prec = float(lines[-3].split(" ")[-2])
    print(prec)
write_result(**{"dataset": "totaltext", "metric":"prec", "eval": prec})