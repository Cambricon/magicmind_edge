import os
import cv2
from PIL import Image
import logging

def voc_dataset(file_list, image_file_path, count):
    with open(file_list, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    current_count = 0
    for line in lines:
        image_name = line.replace("\n", "")
        image_path = os.path.join(image_file_path, image_name + ".jpg")
        img = Image.open(image_path)
        yield img
        current_count += 1
        if current_count > count and count != -1:
            break
class Record:
    def __init__(self, filename):
        self.file = open(os.path.join("output", filename), "w")

    def write(self, line, _print = False):
        self.file.write(line + "\n")
        if _print:
            print(line)

