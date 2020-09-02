import cv2
import numpy as np
import random
import math
import os
import glob

import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
from collections import namedtuple



def clear_folder(folder_path):
    files = glob.glob(folder_path)
    for f in files:
        os.remove(f)
    print("Folder cleared")


def gen_sin_path(img_path, sample):
    map = cv2.imread(img_path)
    x = []
    y = []
    z = []
    for i in range(1, 2 * sample + 1):
        x.append(300 + 8 * i)
        y.append((int(map.shape[0] / 2 + (0.6 * map.shape[0] / 2) * math.sin(math.radians(9 * i)))))
        z.append(0.3 * math.sin(math.radians(20 * i)))
    pt = np.array((x, y, z))
    return pt


def gen_lin_path(img_path, sample):
    map = cv2.imread(img_path)
    # Define starting point
    x_0 = 0.1 * map.shape[1]
    y_0 = 0.1 * map.shape[0]
    start = np.array([x_0, y_0], dtype=np.int32)
    # Define end point
    x_f = np.random.uniform(0.7, 0.9) * map.shape[1]
    y_f = np.random.uniform(0.7, 0.9) * map.shape[0]
    finish = np.array([x_f, y_f], dtype=np.int32)
    x_d = finish[0] - start[0]
    y_d = finish[1] - start[1]
    x = []
    y = []
    z = []
    for i in range(1, sample + 1):
        x.append((int(start[0] + (x_d * i) / sample)))
        y.append((int(start[1] + (y_d * i) / sample)))
        z.append(0.1 * math.sin(math.radians(20 * i)))
    pt = np.array((x, y, z))
    return pt


def gen_elliptical_path(img_path, sample):
    map = cv2.imread(img_path)
    x = []
    y = []
    z = []
    for theta in range(0, 360, int(360 / sample)):
        x.append((int(0.5 * map.shape[1] / 2 * math.cos(math.radians(theta))) + map.shape[1] / 2))
        y.append((int(0.5 * map.shape[0] / 2 * math.sin(math.radians(theta))) + map.shape[0] / 2))
        z.append(0.1 * math.sin(math.radians(8 * theta)))
    pt = np.array((x, y, z))
    return pt


def gen_coll_elliptical_path(img_path, sample, rotation):
    map = cv2.imread(img_path)
    x = []
    y = []
    for theta in range(0, rotation * 360, int(rotation * 360 / sample)):
        lmb = 1 - theta / (rotation * 360)
        x.append((int(lmb * 0.5 * map.shape[1] / 2 * math.cos(math.radians(theta))) + map.shape[1] / 2))
        y.append((int(lmb * 0.5 * map.shape[0] / 2 * math.sin(math.radians(theta))) + map.shape[0] / 2))
    pt = np.array((x, y))
    return pt


def smart_sampler(pt, pixel_x, pixel_y, max_rot_angle, map):
    x = pt[0]
    y = pt[1]
    z = pt[2]
    for i in range(0, len(x)):
        x[i] = max(x[i], pixel_x / 2)
    counter = 0
    med_overlapping = overlapping_area(x,y, pixel_x, pixel_y)
    for i in range(0, len(x)):
        # SCALE - define a scale factor for each dimension
        scale_factor = 1 + z[i]
        pixel_scale_x = int(pixel_x * scale_factor)
        pixel_scale_y = int(pixel_y * scale_factor)
        # ROTATION: Rotate the base image and sample, to rotate a homography matrix is introduced
        rot_angle = np.random.uniform(0, max_rot_angle)
        H_rot = np.array([(math.cos(math.radians(rot_angle)), -math.sin(math.radians(rot_angle)), 0), (math.sin(math.radians(rot_angle)), math.cos(math.radians(rot_angle)), 0), (0, 0, 1)], np.float32)
        map_T = cv2.warpPerspective(map, H_rot, map.shape[:2])

        sample = map_T[int(y[i] - pixel_scale_y / 2):int(y[i] + pixel_scale_y / 2),
                 int(x[i] - pixel_scale_x / 2): int(x[i] + pixel_scale_x / 2)]
        if sample.shape == (pixel_scale_y, pixel_scale_x, 3) and cv2.countNonZero(
                cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)) != 0:
            sample = cv2.resize(sample, (pixel_x, pixel_y))
            counter += 1
            filename = "sample_folder/sample_number" + str(counter) + ".png"
            gps_coordinates(filename, x[i], y[i])
            cv2.imwrite(filename, sample)

    # Plot the map on the 3d graph
    # =============================
    xx, yy = np.meshgrid(np.linspace(0, map.shape[1], map.shape[1]), np.linspace(0, map.shape[0], map.shape[0]))
    X = xx
    Y = yy
    Z = min(z) * np.ones(X.shape) - 2
    # create the figure
    fig = plt.figure()
    # show the 3D rotated projection
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.text2D(0.05, 0.95, "Medium overlapping percentage = " + str(round(med_overlapping, 2)) + "%", transform=ax2.transAxes)
    ax2.plot_surface(X, Y, Z, rstride=10, cstride=10, facecolors=map / 255, shade=False)
    # =============================
    ax2.plot(x, y, z, c='r', marker='o')
    ax2.scatter(x[0], y[0], z[0], s=40, c='b', marker='o')
    plt.show()


def gps_coordinates(filename, x, y):
    with open('gps_xyz.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, x, y])

def overlapping_area(x_c, y_c, pixel_x, pixel_y):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    overlapping_perc = 0
    for i in range(0, len(x_c)):
        ra = Rectangle(x_c[i-1] - pixel_x/2, y_c[i-1] - pixel_y/2, x_c[i-1] + pixel_x/2, y_c[i-1] + pixel_y/2)
        rb = Rectangle(x_c[i] - pixel_x/2, y_c[i] - pixel_y/2, x_c[i] + pixel_x/2, y_c[i] + pixel_y/2)
        dx = min(ra.xmax, rb.xmax) - max(ra.xmin, rb.xmin)
        dy = min(ra.ymax, rb.ymax) - max(ra.ymin, rb.ymin)
        if (dx >= 0) and (dy >= 0):
            overlapping_perc += (dx * dy) * 100 / (pixel_x * pixel_y)
    return overlapping_perc/len(x_c)


if __name__ == "__main__":
    with open('gps_xyz.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_name", "x center", "y center"])
    map = cv2.imread("img_save/V2_map_campus/map_campus_NM_MB.png")
    clear_folder('sample_folder/*')
    # x, y = elliptical_path("img_save/V2_map_campus/map_campus_NM_MB.png", 1)
    smart_sampler(gen_sin_path("img_save/V2_map_campus/map_campus_NM_MB.png", 100), 640, 480, 5, map)
