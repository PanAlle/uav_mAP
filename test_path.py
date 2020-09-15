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


# Clean the sample folder
def clear_folder(folder_path):
    files = glob.glob(folder_path)
    for f in files:
        os.remove(f)
    print("Folder cleared")


# Generate a sine path over the given image
def gen_sin_path(img_path, sample):
    map = cv2.imread(img_path)
    x = []
    y = []
    z = []
    for i in range(1, 2 * sample + 1):
        x.append(300 + map.shape[1] * i / (2 * sample + 1))
        y.append((int(map.shape[0] / 2 + (0.6 * map.shape[0] / 2) * math.sin(math.radians(7 * i)))))
        z.append(0.2 * math.sin(math.radians(20 * i)))
    pt = np.array((x, y, z))
    return pt


# Generate a lienar path over the given image
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


# Generate an elliptical path over the given image
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


# Generate a collapsing ellipses ovet the given image, rotation parameter define the grade of collapsing
def gen_coll_elliptical_path(img_path, sample, rotation):
    map = cv2.imread(img_path)
    x = []
    y = []
    z = []

    for theta in range(0, rotation * 360, int(rotation * 360 / sample)):
        lmb = 1 - theta / (rotation * 360)
        x.append((int(lmb * 0.5 * map.shape[1] / 2 * math.cos(math.radians(theta))) + map.shape[1] / 2))
        y.append((int(lmb * 0.5 * map.shape[0] / 2 * math.sin(math.radians(theta))) + map.shape[0] / 2))
    pt = np.array((x, y))
    return pt


# Get points in pixel based coordinates (px, py, z (z is scale factor and is a number going around 1)) from a csv to sample over a given map

def path_from_data(csv_file):
    x = []
    y = []
    z = []
    with open(csv_file, 'r') as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))
    pt = np.array((x, y, z))
    print("Points vector created")
    return pt


# Creates the samples based on a set of given points, sample wanted dimensions and map
def smart_sampler(pt, pixel_x, pixel_y, map):
    x = pt[0]
    y = pt[1]
    z = pt[2]

    for i in range(0, len(x)):
        x[i] = max(x[i], pixel_x / 2)
    counter = 0
    med_overlapping, fps = overlapping_area(x, y, pixel_x, pixel_y)
    # print("Overlapping area computed")
    for i in range(0, len(x)):
        # SCALE - define a scale factor for each dimension
        scale_factor = z[i]
        pixel_scale_x = int(pixel_x * scale_factor)
        pixel_scale_y = int(pixel_y * scale_factor)
        # Compute the  angle of the tangent considering 2 concurrent points
        if i + 1 != len(x):
            radians = math.atan2(x[i + 1] - x[i], y[i + 1] - y[i])
        else:
            # Since for the last sample it's not possible to compute the tangent it get assigned the last computable
            # value
            radians = math.atan2(y[i] - y[i - 1], x[i] - x[i - 1])
        # Compute and apply the rotation to the full image in order have sample that follow the rotation of the path
        M = cv2.getRotationMatrix2D((x[i], y[i]), math.degrees(- radians), 1)
        map_T = cv2.warpAffine(map, M, (map.shape[1], map.shape[0]))
        # print("Map rotated")
        sample = map_T[int(y[i] - pixel_scale_y / 2):int(y[i] + pixel_scale_y / 2),
                 int(x[i] - pixel_scale_x / 2): int(x[i] + pixel_scale_x / 2)]
        # print("Sample" + str(i) + "computed")
        if sample.shape == (pixel_scale_y, pixel_scale_x, 3) and cv2.countNonZero(
                cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)) != 0:
            sample = cv2.resize(sample, (pixel_x, pixel_y))
            counter += 1
            filename = "sample_folder/sample_number" + str(counter) + ".png"
            gps_coordinates(filename, x[i], y[i])
            cv2.imwrite(filename, sample)
    # Plot the map on the 3d graph
    # =============================
    # xx, yy = np.meshgrid(np.linspace(0, map.shape[1], map.shape[1]), np.linspace(0, map.shape[0], map.shape[0]))
    # X = xx
    # Y = yy
    # Z = min(z) * np.ones(X.shape) - 2
    # create the figure
    # fig = plt.figure()
    # show the 3D rotated projection
    # ax2 = fig.add_subplot(111, projection='3d')
    # ax2.text2D(0.05, 0.95, "Medium overlapping percentage = " + str(round(med_overlapping, 2)) + "%" + ". Req fps: " + str(round(fps, 2)) + "s", transform=ax2.transAxes)
    # # ax2.plot_surface(X, Y, Z, rstride=5, cstride=5, facecolors=map / 255, shade=False)
    # =============================
    # ax2.plot(x, y, z, c='r', marker='o')
    # ax2.scatter(x[0], y[0], z[0], s=40, c='b', marker='o')
    # plt.show()


def gps_coordinates(filename, x, y):
    with open('gps_xyz.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, x, y])


def overlapping_area(x_c, y_c, pixel_x, pixel_y):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    overlapping_perc = 0
    dc = 0
    for i in range(0, len(x_c)):
        ra = Rectangle(x_c[i - 1] - pixel_x / 2, y_c[i - 1] - pixel_y / 2, x_c[i - 1] + pixel_x / 2,
                       y_c[i - 1] + pixel_y / 2)
        rb = Rectangle(x_c[i] - pixel_x / 2, y_c[i] - pixel_y / 2, x_c[i] + pixel_x / 2, y_c[i] + pixel_y / 2)
        dx = min(ra.xmax, rb.xmax) - max(ra.xmin, rb.xmin)
        dy = min(ra.ymax, rb.ymax) - max(ra.ymin, rb.ymin)
        dc += math.hypot(x_c[i] - x_c[i - 1], y_c[i] - y_c[i - 1])
        if (dx >= 0) and (dy >= 0):
            overlapping_perc += (dx * dy) * 100 / (pixel_x * pixel_y)
    speed = 50
    frame = dc / (len(x_c) * speed)
    return overlapping_perc / len(x_c), frame


if __name__ == "__main__":
    with open('gps_xyz.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_name", "x center", "y center"])
    #  GARCHING CAMPUS MAP
    # map = cv2.imread("img_save/V2_map_campus/map_campus_NM_MB.png")

    #  ILLINOIS FIELD MAP
    map = cv2.imread("sample_folder_illinois/illinois_map.png")

    clear_folder('sample_folder/*')
    # smart_sampler(gen_sin_path("img_save/V2_map_campus/map_campus_NM_MB.png", 120), 640, 480, map)
    smart_sampler(path_from_data('flights/393/illinois_sample_infos.csv'), 1000, 750, map)
