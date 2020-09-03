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
print(np.sum(time)/60)
plt.subplot(1, 2, 1)
plt.plot(lenght, kp_full_img, 'b-', label='kp base img')
plt.plot(lenght, kp_next_img,  'r-', label='kp next img')
plt.plot(lenght, good_matches, 'g-', label='good match')
plt.axhline(y=np.sum(good_matches)/len(lenght), color='black',linestyle='--', label= "Med. val good matches = " + str(np.sum(good_matches)/len(lenght)))
plt.xlabel("Samples")
plt.ylabel("Keypoints")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(lenght, time, 'm-.', label='time for each iteration')
plt.axhline(y=round(np.sum(time)/len(lenght), 2), color='black',linestyle='--', label= "Med. val = " + str(round(np.sum(time)/len(lenght), 2)))
plt.xlabel("Tot Samples =" + str(len(lenght)) + " Tot time [m] = " + str(round(np.sum(time)/60, 2)))
plt.ylabel("Time [s]")
plt.legend()
plt.show()