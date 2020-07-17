import csv
import matplotlib.pyplot as plt
import numpy as np
kp_full_img = []
kp_next_img = []
good_matches = []
time = []
lenght = []
with open('csv_plots.csv', 'r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    header = next(csv_reader)
    for row in csv_reader:
        #print(row[0])
        if row == 0:
            pass
        kp_full_img.append(int(row[0]))
        # print(kp_full_img)
        kp_next_img.append(int(row[1]))
        good_matches.append(int(row[2]))
        time.append(float(row[3]))
        lenght.append(len(kp_full_img))
        # print(lenght)

plt.subplot(2, 2, 1)
plt.plot(lenght, kp_full_img, 'b-', label='kp full map')
plt.plot(lenght, kp_next_img,  'r-', label='kp next map')
plt.plot(lenght, good_matches, 'g-', label='good match')

plt.subplot(2, 2, 2)
plt.plot(lenght, time, 'm-.', label='time for each iteration')

plt.subplot(2, 2, 3)
plt.xscale("log")
plt.plot(lenght, kp_full_img, 'b-', label='kp full map')
plt.plot(lenght, kp_next_img,  'r-', label='kp next map')
plt.plot(lenght, good_matches, 'g-', label='good match')

plt.subplot(2, 2, 4)
plt.xscale("log")
plt.plot(lenght, time, 'm-.', label='time for each iteration')

plt.legend()
plt.show()