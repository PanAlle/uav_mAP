import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics



## READ LOG FILE AND MAKE ANOTHER REDUCING SIZE

# IMU_latitude_last = 0
# with open('flights/393/393.log', 'r') as file_read:
#     csv_reader = csv.reader(file_read, delimiter=';')
#     header = next(csv_reader)
#     for row in csv_reader:
#         for i in range(len(row)):
#             if row[i] == 0 or row[i] == math.nan:
#                 next(csv_reader)
#     with open('flights/393/flight_log_uav_mAP.csv', 'a', newline='') as file_write:
#         writer = csv.writer(file_write)
#         writer.writerow([row[9], row[10], row[11]])
#     IMU_latitude_last = row[9]

IMU_latitude = []
IMU_longitude = []
IMU_altitude = []
Euler_yaw = []


with open('flights/393/flight_log_uav_mAP.csv', 'r') as file_read:
    csv_reader = csv.reader(file_read, delimiter=',')
    header1 = next(csv_reader)
    row_number = 0
    for row in csv_reader:
        row_number += 1
        if row_number <= 36000 and row_number % 100 == 0:
            IMU_latitude.append(float(row[0]))
            IMU_longitude.append(float(row[1]))
            IMU_altitude.append(float(row[2]))
    IMU_latitude_median = statistics.mean(IMU_latitude)
    IMU_longitude_median = statistics.mean(IMU_longitude)

    # print(IMU_latitude_median)
    for i in range(len(IMU_latitude)):
        IMU_latitude[i] = (IMU_latitude[i] - IMU_latitude_median)
    for i in range(len(IMU_longitude)):
        IMU_longitude[i] = (IMU_longitude[i] - IMU_longitude_median)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(IMU_latitude, IMU_longitude, IMU_altitude)
plt.scatter(IMU_latitude, IMU_longitude, s=1)
plt.show()