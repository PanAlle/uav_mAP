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
    return [atoi(c) for c in re.split(r"(\d+)", text)]

def load_images_from_folder(folder):
    images = []
    sorted_list = []
    for filename in os.listdir(folder):
        sorted_list.append(filename)
    sorted_list.sort(key=natural_keys)
    # sorted_list.reverse()
    for filename in sorted_list:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    print("Images loaded")
    return images

def findDimensions(image, H_inv, H):
    # Initialize 1x3 array of 1 to hold x,y,channel
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)
    # Get image Height and Width
    (y, x) = image.shape[:2]
    # print(np.linalg.det(H[:2, :2])/math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2))
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
        # Normalize the coordinates in order to define a point in homogeneous coordinates.
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

    # Limit to zero the maximum value of min_x and min_y
    min_x = min(0, min_x)
    min_y = min(0, min_y)
    return (min_x, min_y, max_x, max_y)

def features_detection(img):
    descriptor = cv2.xfeatures2d.SURF_create()
    # Compute keypoint and relative descriptor, None mask applied to the images
    kp_pt, kp_descriptor = descriptor.detectAndCompute(img, None)
    return kp_pt, kp_descriptor

def features_detection_next(img, mask_1):
    descriptor = cv2.xfeatures2d.SURF_create()
    # Compute keypoint and relative descriptor, None mask applied to the images
    # ret, mask = cv2.threshold(mask_1, 0, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("full_image_warp", cv2.resize(cv2.bitwise_and(img, mask), (640,480)))
    # cv2.waitKey(30)
    print(mask.shape)
    kp_pt, kp_descriptor = descriptor.detectAndCompute(img, mask)
    return kp_pt, kp_descriptor

def feature_matching(base_img_descriptor, next_img_descriptor):
    # Create a Brute Force matcher object, for SURF is advised to use k nearest neighbour and define 2 groups.
    bf = cv2.BFMatcher()
    # Returns the 2 best matches for each descriptor
    matches = bf.knnMatch(base_img_descriptor, next_img_descriptor, k=2)
    # In order to only consider the points that are actually matching a ratio is defined (Lowe's ratio), when the
    # first distance between the keypoints is significantly smaller than the second one
    # (2 distances coming from knn), then the two keypoitns are considered as matching and used to homography.
    # Otherwise, since there is no security of which between the two best matches is the keypoints corresponded
    # to, the current match is discarded.
    # Initialize an array to store selceted matches based on Lowe's ratio
    sel_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            sel_matches.append(m)
    return sel_matches


def find_homography(kp_base_img, kp_next_img, sel_matches):
    src_pts = np.float32([kp_base_img[m.queryIdx].pt for m in sel_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_next_img[m.trainIdx].pt for m in sel_matches]).reshape(-1, 1, 2)
    # Find homography matrix wiht RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("No Homography")
    else:
        return H


def stitching(full_img, next_img, H, vector, offset_value):
    # Normalize to maintaining homogeneous coordinate system
    H = H / H[2, 2]

    print(H)
    # Inverse homography, from the next image frame to the base image frame
    H_inv = linalg.inv(H)
    # Find the rectangular hull containing the next_img and the next_image, since a new reference frame is defined then
    # both base and next image need to be translated
    (min_x, min_y, max_x, max_y) = findDimensions(next_img,  H_inv, H)
    # Adjust max_x, max_y in order to include also the first image
    max_x = max(max_x, full_img.shape[1])
    max_y = max(max_y, full_img.shape[0])
    # Define the move vector for each pixel
    move_h = np.identity(3, np.float64)
    # Translate the origin of the image in the positive part of the plane, in order to see it
    if min_x < 0 or min_y < 0:
        move_h = H
        print("Move H", H)
        move_h[0, 2] = 0
        move_h[1, 2] = 0
        move_h[0, 2] += -min_x
        max_x += -min_x
        move_h[1, 2] += -min_y
        max_y += -min_y
        img_w = int(math.ceil(max_x) * math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2))
        img_h = int(math.ceil(max_y) * np.linalg.det(H[:2, :2]) / math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2))
    else:
        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))
    # First the second image need to be taken to the first image reference frame (H inv) and then need to be
    # translated by move_h. The two transformation can be coupled in mod_inv_h
    mod_inv_h = np.matmul(H_inv, move_h)
    # Return the closest integer near a given number
    print(img_h, img_w)

    next_img_center_point = np.identity(3, np.float32)
    next_img_center_point[0, 2] = (next_img.shape[1]/2)
    next_img_center_point[1, 2] = (next_img.shape[0]/2)
    # print("Next image center point", next_img_center_point,"\n")
    next_img_center_point = np.matmul(H_inv, next_img_center_point)
    # print("Center point X:", next_img_center_point[0,2],"Y:", next_img_center_point[1,2], "\n")

    pts = np.array(next_img_center_point)
    vector.append(pts)
    # Warp the new image given the homograph from the old image
    for i in range(0, len(vector)):
        vector[i] = np.dot(vector[i], move_h)

    last_img_x_center = int(vector[-1][0][2])
    last_img_y_center = int(vector[-1][1][2])

    edge_1 = np.array([[last_img_x_center - int(next_img.shape[1] / 2) - offset_value, last_img_y_center - int(next_img.shape[0] / 2) - offset_value]])
    edge_2 = np.array([[last_img_x_center + int(next_img.shape[1] / 2) + offset_value, last_img_y_center - int(next_img.shape[0] / 2) - offset_value]])
    edge_3 = np.array([[last_img_x_center + int(next_img.shape[1] / 2) + offset_value, last_img_y_center + int(next_img.shape[0] / 2) + offset_value]])
    edge_4 = np.array([[last_img_x_center - int(next_img.shape[1] / 2) - offset_value, last_img_y_center + int(next_img.shape[0] / 2) + offset_value]])

    edges = np.array([edge_1, edge_2, edge_3, edge_4])
    print(move_h)
    # print(H)
    cv2.imshow("full_imag", cv2.resize(full_img, (640,480)))
    cv2.waitKey(30)
    full_img_warp = cv2.warpPerspective(full_img, move_h, (img_w, img_h))
    cv2.imshow("full_image_warp", cv2.resize(full_img_warp, (640,480)))
    cv2.waitKey(30)
    next_img_warp = cv2.warpPerspective(next_img, mod_inv_h, (img_w, img_h))
    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)


    # Create a mask from the warped image for constructing masked composite (insert black
    # base on next image, covering the first one)
    (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    enlarged_full_img = cv2.add(enlarged_base_img, full_img_warp, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
    final_img = cv2.add(enlarged_full_img, next_img_warp, dtype=cv2.CV_8U)

    # Add the warped image with 8bit/pixel (0 - 255)
    mask_1 = np.zeros(final_img.shape, dtype=np.uint8)
    cv2.fillPoly(mask_1, pts=[edges], color=(255, 255, 255))
    maksed_image = cv2.bitwise_and(final_img, mask_1)

    return final_img, maksed_image, vector, mask_1


if __name__ == "__main__":
    with open('../csv_plots.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SURF_kp full_img", "SURF_kp next_img", "number of good matches"])
    images = load_images_from_folder("sample_folder")
    base_img = images[0]
    # huge_image = np.zeros((int(0.4 * images[0].shape[0]*len(images)), int(0.4 * images[0].shape[1]*len(images)), 3), np.uint8)
    # huge_image[int(huge_image.shape[0]/2): int(huge_image.shape[0]/2) + base_img.shape[0], int(huge_image.shape[1]/2): int(huge_image.shape[1]/2) + base_img.shape[1]] = base_img
    # base_img = huge_image
    print("First iteration completed")
    base_img_center_point = np.identity(3, np.float32)
    base_img_center_point[0, 2] = base_img.shape[1] / 2
    base_img_center_point[1, 2] = base_img.shape[0] / 2
    base_pts = np.array(base_img_center_point)
    offset_value = 100
    vector = [base_pts]
    for i in range(1, len(images)):
        start_time = time.time()
        next_img = images[i]
        if i == 1:
            img1_GS = cv2.GaussianBlur(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            kp_base_img, kp_descriptor_base_img = features_detection(img1_GS)
            kp_next_img, kp_descriptor_next_img = features_detection(img2_GS)
        else:
            img1_GS = cv2.GaussianBlur(cv2.cvtColor(neg, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            kp_base_img, kp_descriptor_base_img = features_detection_next(img1_GS, mask_1)
            kp_next_img, kp_descriptor_next_img = features_detection(img2_GS)
        sel_matches = feature_matching(kp_descriptor_base_img, kp_descriptor_next_img)
        H = find_homography(kp_base_img, kp_next_img, sel_matches)
        # print("Current homography matrix scale introduced on X: " + str(100*(math.sqrt(H[0,0]**2 + H[1,0]**2) - 1)) + " Y:" + str(100 * ((np.linalg.det(H[:2, :2])/math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2) -1))) + "\n")
        if i == 1:
            final_img, neg, next_center, mask_1 = stitching(base_img, next_img, H, vector, offset_value)
        else:
            final_img, neg, next_center, mask_1 = stitching(final_img, next_img, H, vector, offset_value)

        # print(next_center[1])
        base_img = final_img

        with open('../csv_plots.csv', 'a', newline ='') as file:
            writer = csv.writer(file)
            writer.writerow([len(kp_base_img), len(kp_next_img), len(sel_matches), (time.time() - start_time)])
        print("iteration number ", i, " completed")
    final_img = cv2.medianBlur(final_img, 3)
    for i in next_center:
        cv2.circle(final_img, (int(i[0, 2]), int(i[1, 2])), 5,  (255, 255, 0), -1)
    row_of_interest = []
    with open('../gps_xyz.csv', 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            row_of_interest.append(row)
    for i in range (0, len(next_center)):
        if i % 5 == 0:
            # print(row_of_interest[i])
            cv2.putText(final_img, "X: " + str(row_of_interest[i][1]) + " Y: " + str(row_of_interest[i][2]), (int(next_center[i][0][2] +10), int(next_center[i][1][2])), cv2.FONT_ITALIC, 0.5 ,(255, 255, 0) )
    cv2.imshow("next", final_img)
    save_file_name = "img_save/Linear_sine_zoom_test/REVREV_Test1_Pix" + str(images[0].shape[0]) + "X" + str(images[0].shape[1]) + "_N_img" + str(len(images)) + "_Offset_" + str(offset_value) +".png"
    cv2.imwrite(save_file_name, final_img)
    cv2.waitKey()



