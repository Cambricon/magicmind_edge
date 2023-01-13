import os
import cv2
import tqdm
import numpy as np
import pdb
import json, argparse

def get_tusimple_list(root, label_list):
    '''
    Get all the files' names from the json annotation
    '''
    label_json_all = []
    for l in label_list:
        l = os.path.join(root,l)
        label_json = [json.loads(line) for line in open(l).readlines()]
        label_json_all += label_json
    names = [l['raw_file'] for l in label_json_all]
    h_samples = [np.array(l['h_samples']) for l in label_json_all]
    lanes = [np.array(l['lanes']) for l in label_json_all]

    line_txt = []
    for i in range(len(lanes)):
        line_txt_i = []
        for j in range(len(lanes[i])):
            if np.all(lanes[i][j] == -2):
                continue
            valid = lanes[i][j] != -2
            line_txt_tmp = [None]*(len(h_samples[i][valid])+len(lanes[i][j][valid]))
            line_txt_tmp[::2] = list(map(str,lanes[i][j][valid]))
            line_txt_tmp[1::2] = list(map(str,h_samples[i][valid]))
            line_txt_i.append(line_txt_tmp)
        line_txt.append(line_txt_i)

    return names,line_txt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the Tusimple dataset')
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()
    # testing set
    names,line_txt = get_tusimple_list(args.root, ['test_tasks_0627.json'])
    # generate testing set for testing
    with open(os.path.join(args.root,'test.txt'),'w') as fp:
        for name in names:
            fp.write(name + '\n')

