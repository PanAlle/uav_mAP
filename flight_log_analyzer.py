import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
import utm
import cv2


# ==================
# This .py file is created to read the log from the UAV.

# Linear interpolation function to pass from the lat, long to pixel_x, pixel_y
def linear_interpolation(x1, y1, x2, y2, x3):
    slope = (y2 - y1) / (x2 - x1)
    y3 = (slope * (x3 - x1)) + y1
    return y3


# Read log file and copy th column of interest to another csv file

with open('flights/393/393.log', 'r') as file_read:
    csv_reader = csv.reader(file_read, delimiter=';')
    header = next(csv_reader)
    for row in csv_reader:
        for i in range(len(row)):
            if row[i] == 0 or row[i] == math.nan:
                next(csv_reader)
        with open('flights/393/flight_log_uav_mAP.csv', 'a', newline='') as file_write:
            writer = csv.writer(file_write)
            writer.writerow([row[9], row[10], row[11], row[3]])

IMU_latitude = []
IMU_longitude = []
IMU_altitude = []
Euler_yaw = []
# Read the file in which the columns of interest are saved
with open('flights/393/flight_log_uav_mAP.csv', 'r') as file_read:
    csv_reader = csv.reader(file_read, delimiter=',')
    header1 = next(csv_reader)
    row_number = 0
    # GOOGLE MAPS CSV FILE
    # Initialize csv file that can be used to save create custom map points on google maps
    # with open('flights/393/csv_google.csv', 'w', newline='') as file_write_1:
    #     writer = csv.writer(file_write_1)
    #     writer.writerow(["Latitude", "Longitude", "Point" + str(row_number)])
    hour_offset = 1440000
    for row in csv_reader:
        row_number += 1
        # The log file contains 100 reads per second, 100 rows = 1 second. Read from the first hour (60*60*100) for a
        # minute. Read from the first hour on in order to avoid reads related to takeoff
        if row_number >= hour_offset + 375000 and row_number <= hour_offset + 397000 and row_number % 50 == 0:
            IMU_latitude.append(float(row[0]))
            IMU_longitude.append(float(row[1]))
            IMU_altitude.append(float(row[2]))
            Euler_yaw.append(float(row[3]))
            # GOOGLE MAPS CSV FILE
            # if row_number % 100 == 0:
            # with open('flights/393/csv_google.csv', 'a', newline='') as file_write_1:
            #     writer = csv.writer(file_write_1)
            #     writer.writerow([row[0], row[1], "Point" + str(row_number)])
    IMU_altitude_mean = statistics.mean(IMU_altitude)
    # Initialize csv to save the points that are going to be used to define a path and sample frames.
    with open('flights/393/illinois_sample_infos.csv', 'w') as sample_write:
        writer = csv.writer(sample_write, delimiter=',')
        writer.writerow(["X", "Y", "Scale", "Rotation"])
    x = []
    y = []
    with open('flights/393/illinois_sample_infos.csv', 'a') as sample_write:
        writer = csv.writer(sample_write, delimiter=',')
        for i in range(len(IMU_latitude)):
            # The value for the interpolation are manually inserted as follow:
            # linear_interpolation(lat1, px1, lat2, px2) and linear_interpolation(long1, py1, long2, py2)
            IMU_latitude[i] = linear_interpolation(40.060875, 8500, 40.056580, 1300, IMU_latitude[i])
            IMU_longitude[i] = linear_interpolation(-88.552345, 10000, -88.547112, 1400, IMU_longitude[i])
            x.append(IMU_latitude[i])
            y.append(IMU_longitude[i])
            # The height value is weighted on the mean value of the height in order to have a value ranging around 1
            scale = IMU_altitude[i] / IMU_altitude_mean
            print(IMU_altitude_mean)
            writer.writerow([IMU_latitude[i], IMU_longitude[i], scale, Euler_yaw[i]])
# Plot the points on the image
im = plt.imread("illinois_map_v1.png")
implot = plt.imshow(im)
plt.scatter(x, y, s=10)
plt.show()
