import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
import utm
import cv2

# ==================
# This .py file is created to read the log from the UAV.

# =Linear interpolation function to pass from the lat, long to pixel_x, pixel_y
def translate_val(x1, y1, x2, y2, x3):
    slope = (y2 - y1) / (x2 - x1)
    y3 = (slope * (x3 - x1)) + y1
    return y3


# Read log file and copy th column of interest to another csv file

# with open('flights/393/393.log', 'r') as file_read:
#     csv_reader = csv.reader(file_read, delimiter=';')
#     header = next(csv_reader)
#     for row in csv_reader:
#         for i in range(len(row)):
#             if row[i] == 0 or row[i] == math.nan:
#                 next(csv_reader)
#         with open('flights/393/flight_log_uav_mAP.csv', 'a', newline='') as file_write:
#             writer = csv.writer(file_write)
#             writer.writerow([row[9], row[10], row[11], row[3]])


IMU_latitude = []
IMU_longitude = []
IMU_altitude = []
Euler_yaw = []
# Reaf the file in which the columns of interest are saved
with open('flights/393/flight_log_uav_mAP.csv', 'r') as file_read:
    csv_reader = csv.reader(file_read, delimiter=',')
    header1 = next(csv_reader)
    row_number = 0
    # Initialize csv file that can be used to save create custom map points on google maps
    # with open('flights/393/csv_google.csv', 'w', newline='') as file_write_1:
    #     writer = csv.writer(file_write_1)
    #     writer.writerow(["Latitude", "Longitude", "Point" + str(row_number)])
    for row in csv_reader:
        row_number += 1
        # read from the first hour (60*60*100) where 100 are sample per seconds for a minute.
        if row_number >= 360000 and row_number <= 365000 and row_number % 100 == 0:
            IMU_latitude.append(float(row[0]))
            IMU_longitude.append(float(row[1]))
            IMU_altitude.append(float(row[2]))
            Euler_yaw.append(float(row[3]))
            # if row_number % 100 == 0:
                # with open('flights/393/csv_google.csv', 'a', newline='') as file_write_1:
                #     writer = csv.writer(file_write_1)
                #     writer.writerow([row[0], row[1], "Point" + str(row_number)])
    IMU_altitude_mean = statistics.mean(IMU_altitude)
    # Initialize csv to be used to sample
    with open('flights/393/illinois_sample_infos.csv', 'w') as sample_write:
        writer = csv.writer(sample_write,  delimiter=',')
        writer.writerow(["X", "Y", "Scale", "Rotation"])
    # Row number should be apprx. 2774999
    print(row_number)
    x = []
    y = []
    with open('flights/393/illinois_sample_infos.csv', 'a') as sample_write:
        writer = csv.writer(sample_write, delimiter=',')
        for i in range(len(IMU_latitude)):
            # The value for the interpolation are manually read
            # translate_val(lat1, px1, lat2, px2) and translate_val(log1, py1, log2, py2)
            IMU_latitude[i] = translate_val(40.060875, 9000, 40.056580, 1800, IMU_latitude[i]) - 300
            IMU_longitude[i] = translate_val(-88.552345, 10500, -88.547112, 1900, IMU_longitude[i])
            x.append(IMU_latitude[i] - 300)
            y.append(IMU_longitude[i])
            scale = IMU_altitude[i] / IMU_altitude_mean
            writer.writerow([IMU_latitude[i], IMU_longitude[i], scale, Euler_yaw[i]])
# Plot the points on the image
im = plt.imread("sample_folder_illinois/illinois_map.png")
implot = plt.imshow(im)
plt.scatter(x, y, s=10)
plt.show()
