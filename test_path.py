import cv2
import numpy as np
import random
import math
import os
import glob

def clear_folder(folder_path):
    files = glob.glob(folder_path)
    for f in files:
        os.remove(f)
    print("Folder cleared")
def gen_path(img_path):
    map = cv2.imread(img_path)
    h, w, _ = map.shape
    # Define starting point
    x_0 = random.uniform(0, 1) * w
    y_0 = random.uniform(0, 0.3) * h
    start = np.array([x_0, y_0], dtype=np.int32)
    print(x_0, y_0)
    # Define end point
    x_f = x_0
    y_f = random.uniform(0.7, 0.9) * h
    finish = np.array([x_f, y_f], dtype=np.int32)
    return start, finish


def sampler(start, finish, pixel, img_path):
    print(start, finish)
    map = cv2.imread(img_path)
    print(map.shape)
    for i in range(0, len(start)):
        start[i] = max(start[i], pixel/2)
    # considering images with Pixel*Pixel size on a straight line and a overlapping of 80perc the step form a image to the other is 0.2*pixel
    step = int(0.2 * pixel)
    dist = np.linalg.norm(finish - start)
    pos = start
    counter = 0
    for i in range(0, int(dist / step)):
        sample = map[int(pos[0] - pixel / 2): int(pos[0] + pixel / 2), int(pos[1] - pixel / 2):int(pos[1] + pixel / 2)]
        if sample.shape != (pixel, pixel, 3):
            print("Returned on border")
            return
        if cv2.countNonZero(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)) != 0:
            counter += 1
            print("non black")
            filename = "sample_folder/sample_number" + str(counter) + ".png"
            cv2.imwrite(filename, sample)
        for i in range(0, len(pos)):
            pos[i] += step
            print(pos[i])

def read_map(img_path):
    map = cv2.imread(img_path)
    h, w, _ = map.shape


if __name__ == "__main__":
    clear_folder('sample_folder/*')
    start, finish = gen_path("img_save/V2_map_campus/map_campus_NM_MB.png")
    sampler(start, finish, 300, "img_save/V2_map_campus/map_campus_NM_MB.png")