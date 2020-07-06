import cv2
import numpy as np


def features_detection(img, type):
    if type == "SURF":
        descriptor = cv2.xfeatures2d.SURF_create()
        #Other than none we can pass a mask to cover parts of the image that is not needed
        key_pts, features = descriptor.detectAndCompute(img, None)
    elif type == "ORB":
        descriptor = cv2.ORB_create(nfeatures=10000)
        #Other than none we can pass a mask to cover parts of the image that is not needed
        key_pts, features = descriptor.detectAndCompute(img, None)
    return key_pts, features


img = cv2.imread("img_folder/Screenshot from 2020-04-08 22-55-52.png", cv2.IMREAD_GRAYSCALE)
kp, p = features_detection(img, "ORB")
img_mod = cv2.drawKeypoints(img, kp, None)
cv2.imshow("keypoint", img_mod)
print(p)
cv2.waitKey()
