import cv2
import numpy as np
import random
import math
import os
import glob
import matplotlib.pyplot as plt
import csv



def clear_folder(folder_path):
    files = glob.glob(folder_path)
    for f in files:
        os.remove(f)
    print("Folder cleared")


def gen_lin_path(img_path, sample, pixel, flag):
    map = cv2.imread(img_path)
    # Define starting point
    x_0 = random.uniform(0, 1) * map.shape[1]
    y_0 = random.uniform(0, 0.3) * map.shape[0]
    start = np.array([x_0, y_0], dtype=np.int32)
    # Define end point
    x_f = random.uniform(0.7, 0.9) * map.shape[1]
    y_f = random.uniform(0.7, 0.9) * map.shape[0]
    finish = np.array([x_f, y_f], dtype=np.int32)
    x_d = finish[0] - start[0]
    y_d = finish[1] - start[1]
    x = []
    y = []
    for i in range(1, sample+1):
        x.append((int(start[0] + (x_d * i)/sample)))
        y.append((int(start[1] + (y_d * i) / sample)))
    if flag == -1:
        start[0] += pixel/2
        finish[0] += pixel/2
        x_d = finish[0] - start[0]
        y_d = finish[1] - start[1]
        for i in range(1, sample+1):
            x.append((int(finish[0] - (x_d * i)/sample)))
            y.append((int(finish[1] - (y_d * i) / sample)))
    pt = np.array((x, y))
    plt.scatter(x,y)
    plt.show()
    return pt


def gen_elliptical_path(img_path, sample):
    map = cv2.imread(img_path)
    x = []
    y = []
    z = []
    for theta in range(0, 360, int(360/sample)):
        x.append((int(0.5 * map.shape[1] / 2 * math.cos(math.radians(theta))) + map.shape[1] / 2))
        y.append((int(0.5 * map.shape[0] / 2 * math.sin(math.radians(theta))) + map.shape[0] / 2))
        z.append(0.02*math.sin(math.radians(8*theta)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()
    pt = np.array((x, y, z))
    return pt

def gen_coll_elliptical_path(img_path, sample, rotation):
    map = cv2.imread(img_path)
    x = []
    y = []
    # print(pt)
    for theta in range(0, rotation*360, int(rotation*360/sample)):
        lmb = 1 - theta / (rotation*360)
        x.append((int(lmb * 0.5 * map.shape[1] / 2 * math.cos(math.radians(theta))) + map.shape[1] / 2))
        y.append((int(lmb * 0.5 * map.shape[0] / 2 * math.sin(math.radians(theta))) + map.shape[0] / 2))
    pt = np.array((x, y))
    plt.scatter(x,y)
    plt.show()
    return pt

def smart_sampler(pt, pixel_x, pixel_y, map):
    x = pt[0]
    y = pt[1]
    z = pt[2]
    for i in range(0, len(x)):
        x[i] = max(x[i], pixel_x / 2)
    counter = 0
    for i in range(0, len(x)):
        scale_factor = 1 + z[i]
        pixel_scale_x = int(pixel_x * scale_factor)
        pixel_scale_y = int(pixel_y * scale_factor)
        sample = map[int(y[i] - (pixel_scale_y) / 2):int(y[i] + (pixel_scale_y) / 2), int(x[i] - (pixel_scale_x) / 2): int(x[i] + (pixel_scale_x) / 2)]
        if sample.shape == (pixel_scale_y, pixel_scale_x, 3) and cv2.countNonZero(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)) != 0:
            sample = cv2.resize(sample, (pixel_x, pixel_y))
            counter += 1
            filename = "sample_folder/sample_number" + str(counter) + ".png"
            gps_coordinates(filename, x[i], y[i])
            cv2.imwrite(filename, sample)

def gps_coordinates(filename, x,y):
    with open('gps_xyz.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, x, y ])


if __name__ == "__main__":
    with open('gps_xyz.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_name", "x center", "y center"])
    map = cv2.imread("img_save/V2_map_campus/map_campus_NM_MB.png")
    clear_folder('sample_folder/*')
    #x, y = elliptical_path("img_save/V2_map_campus/map_campus_NM_MB.png", 1)
    smart_sampler(gen_elliptical_path("img_save/V2_map_campus/map_campus_NM_MB.png", 100), 640, 480, map)