import os
import numpy as np
import argparse
import sys
sys.path.append(os.path.join(os.getenv("MAGICMIND_EDGE"), "utils"))
from record_result import write_result
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", dest = 'result_file', help = "result txt", type = str) 

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    tot_correct = 0
    tot_count = 0
    wrong_cases = []
    fileHandler  =  open  (args.result_file,  "r")
    for line in fileHandler:  
        real = line.split(' ')[0].split('_')[1].lower()
        pred = line.split(' ')[-1].strip()
        tot_count += 1
        if  pred == real:
            tot_correct += 1
        else:
            wrong_cases.append((real, pred))
        # print(real)
        # print(pred)
    
    acc = float(tot_correct) / float(tot_count)
    # print("wrong_cases",wrong_cases)
    print("==================== Results ====================")
    print("accuracy: %f"%(acc))
    print("==================== Results ====================")
    write_result(**{"dataset": "mjsynth", "metric":"acc", "eval": acc})