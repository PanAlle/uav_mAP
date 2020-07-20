import cv2
import numpy as np
import math
from numpy import linalg
import os
import re
import csv
import time


# Sort images based on number
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_images_from_folder(folder):
    images = []
    sorted_list = []
    for filename in os.listdir(folder):
        sorted_list.append(filename)
    sorted_list.sort(key=natural_keys)
    for filename in sorted_list:
        img = cv2.imread(os.path.join(folder,filename))
        #img = img[:][230:1750]
        print(filename)
        if img is not None:
            images.append(img)
    print("Images loaded")
    return images

def findDimensions(image, H_inv):
    # Initialize 1x3 array of 1 to hold x,y,channel
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    # Get image Height and Width
    (y, x) = image.shape[:2]

    # Store corner points
    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    # Initialize variables for max and min, in order to define overall image frame
    max_x = None
    max_y = None
    min_x = None
    min_y = None

    # For each one of the points we need to apply the inverse Homography matrix in order to transform the points of the next image to the base frame
    for pt in [base_p1, base_p2, base_p3, base_p4]:

        # Transform next_image corners into the base space
        hp = np.matmul(np.array(H_inv, np.float32), np.array(pt, np.float32).reshape(-1,1))

        # Normalize the coordinates, by defining the correct point the output is a line vector
        normal_pt = np.array([hp[0, 0] / hp[2, 0], hp[1, 0] / hp[2, 0]], np.float32)
        # In order to define a rectangular hull containing the image, keep only min and max values of x,y
        if (max_x == None or normal_pt[0] > max_x):
            max_x = normal_pt[0]

        if (max_y == None or normal_pt[1] > max_y):
            max_y = normal_pt[1]

        if (min_x == None or normal_pt[0] < min_x):
            min_x = normal_pt[0]

        if (min_y == None or normal_pt[1] < min_y):
            min_y = normal_pt[1]
    # Limit to zero the value of min_x and min_y
    min_x = min(0, min_x)
    min_y = min(0, min_y)
    return (min_x, min_y, max_x, max_y)


def features_detection(img):
    descriptor = cv2.xfeatures2d.SURF_create()
    # None stand for no mask applied
    kp_pt, kp_descriptor = descriptor.detectAndCompute(img, None)
    return kp_pt, kp_descriptor


def feature_matching(base_img_descriptor, next_img_descriptor):
    # Create a Brute Force matcher object, for SURF is advised to use k nearest neighbour and define 2 groups.
    bf = cv2.BFMatcher()
    #give the two best matches for each descriptor
    matches = bf.knnMatch(base_img_descriptor, next_img_descriptor, k=2)

    # Initialize an array to store good matches in terms of distance
    sel_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            sel_matches.append(m)
    return sel_matches


def find_homography(kp_base_img, kp_next_img, sel_matches):
    src_pts = np.float32([kp_base_img[m.queryIdx].pt for m in sel_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_next_img[m.trainIdx].pt for m in sel_matches]).reshape(-1, 1, 2)
    # Find homography matrix and do perspective transform by using of RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("No Homography")
    else:
        return H


def stitching(full_img, next_img, H):
    # Normalize to maintaining homogeneous coordinate system
    H = H / H[2, 2]
    # Inverse matrix
    H_inv = linalg.inv(H)
    # Find the rectangular hull containing the next_img in the base_img reference frame
    (min_x, min_y, max_x, max_y) = findDimensions(next_img,  H_inv)
    # Adjust max_x, max_y in order to include also the first image
    max_x = max(max_x, full_img.shape[1])
    max_y = max(max_y, full_img.shape[0])
    # Define the move vector for each pixel
    move_h = np.identity(3, np.float32)
    # Translate the origin of the image in the positive part of the plane, in order to see it
    if (min_x < 0):
        move_h[0, 2] += -min_x
        max_x += -min_x

    if (min_y < 0):
        move_h[1, 2] += -min_y
        max_y += -min_y
    # First the second image need to be taken to the first image reference frame (H inv) and then need to be
    # translated by move_h. The two transformation can be coupled in mod_inv_h
    mod_inv_h = np.matmul(H_inv, move_h)
    # Return the closest integer near a given number
    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

    # Warp the new image given the homography from the old image
    full_img_warp = cv2.warpPerspective(full_img, move_h, (img_w, img_h))
    next_img_warp = cv2.warpPerspective(next_img, mod_inv_h, (img_w, img_h))
    # Create a black image of max required dimensions
    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

    # Create a mask from the warped image for constructing masked composite (insert black
    # base on next image, covering the first one)
    (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    enlarged_full_img = cv2.add(enlarged_base_img, full_img_warp, mask=np.bitwise_not(data_map),dtype=cv2.CV_8U)
    # Add the warped image with 8bit/pixel (0 - 255)
    final_img = cv2.add(enlarged_full_img, next_img_warp, dtype=cv2.CV_8U)
    return final_img, next_img_warp

if __name__ == "__main__":
    with open('csv_plots.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SURF_kp full_img", "SURF_kp next_img", "number of good matches"])

    images = load_images_from_folder("sample_folder")
    move_h = np.identity(3, np.float32)
    base_img = images[0]
    for i in range(1, len(images)):
        start_time = time.time()
        next_img = images[i]
        if (i == 1):
            img1_GS = cv2.GaussianBlur(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        else:
            img1_GS = cv2.GaussianBlur(cv2.cvtColor(neg, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        print("At iteration", i, "applied Gaussian blur")
        kp_base_img, kp_descriptor_base_img = features_detection(img1_GS)
        kp_next_img, kp_descriptor_next_img = features_detection(img2_GS)
        print("At iteration", i, "applied computed descriptor")
        sel_matches = feature_matching(kp_descriptor_base_img, kp_descriptor_next_img)
        print("At iteration", i, "applied Sorted Matches")
        H = find_homography(kp_base_img, kp_next_img, sel_matches)
        if (i == 1):
            final_img, neg = stitching(base_img, next_img, H)
        else:
            final_img, neg = stitching(final_img, next_img, H)

        print("At iteration", i, "applied compute old homograhy")
        base_img = final_img
        with open('csv_plots.csv', 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow([len(kp_base_img), len(kp_next_img), len(sel_matches), (time.time() - start_time)])

        print("done")

    cv2.imshow("next", final_img)
    print(final_img.shape)
    cv2.imwrite("img_save/sample_path_test_1.png", cv2.medianBlur(final_img, 3))
    cv2.waitKey()