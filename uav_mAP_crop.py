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


# In the image name only consider the number for sorting, no text part.
def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


# Load images from a certain folder and insert them in list
def load_images_from_folder(folder):
    images = []
    sorted_list = []
    for filename in os.listdir(folder):
        sorted_list.append(filename)
    sorted_list.sort(key=natural_keys)
    # Analyze the images in starting from the last one. Used for test purpose
    # sorted_list.reverse()
    for filename in sorted_list:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    print("Images loaded")
    return images


# Given the image in the base frame it computes the rectangular hull containing the image in the full image frame.
def find_dimensions(image, H_inv):
    # Initialize 1x3 array of 1 to hold x,y,channel
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)
    # Get image Height and Width
    (y, x) = image.shape[:2]
    # Store corner points of the image
    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    # Initialize variables for max and min, in order to define hull dimensions
    max_x = None
    max_y = None
    min_x = None
    min_y = None

    # For each one of the points we need to apply the inverse Homography matrix in order to transform the points of
    # the image to the full image frame
    for pt in [base_p1, base_p2, base_p3, base_p4]:
        hp = np.matmul(np.array(H_inv, np.float32), np.array(pt, np.float32).reshape(-1, 1))
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

    return (min_x, min_y, max_x, max_y)


# Compute key points and the relative descriptors for the given image based on SURF
def features_detection(img):
    descriptor = cv2.xfeatures2d.SURF_create()
    # Compute keypoint and relative descriptor, None mask applied to the images
    kp_pt, kp_descriptor = descriptor.detectAndCompute(img, None)
    return kp_pt, kp_descriptor


# Define the matching key points between two concurrent images based on the analysis of the descriptors
def feature_matching(base_img_descriptor, next_img_descriptor):
    # Create a Brute Force matcher object
    bf = cv2.BFMatcher()
    # For SURF is advised to use k nearest neighbour and define 2 groups, speaking in the key points language this
    # mean that each key point is associated with 2 other ones, which have the "closest" descriptor to the first key
    # point.
    matches = bf.knnMatch(base_img_descriptor, next_img_descriptor, k=2)
    # In order to only consider the points that are actually matching a ratio is defined (Lowe's ratio),
    # when the distance between the key point and the first matching one is significantly smaller than the distance
    # from the second matching one, then the two key points are considered as matching and used to compute homography.
    # Otherwise, since there is no certainty on which between the two best matches is the key point corresponded to,
    # the current match is discarded.
    # Initialize an array to store selected matches based on Lowe's ratio
    sel_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            sel_matches.append(m)
    return sel_matches


# Compute teh homograpyh matrix based on the RANSAC algorithm
def find_homography(kp_base_img, kp_next_img, sel_matches):
    src_pts = np.float32([kp_base_img[m.queryIdx].pt for m in sel_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_next_img[m.trainIdx].pt for m in sel_matches]).reshape(-1, 1, 2)
    # Find homography matrix wiht RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("No Homography")
    else:
        return H


# Stitch the images together
# - H, holography matrix transformation from full image frame to new image frame
# - H_inv, homography matrix transformation from new image frame to full image frame.
# - H_trans, homography matrix transformation from the new image cropped frame to the new image frame
def stitching(full_img, next_img, H, H_trans, vector, offset_value):
    # Normalize to maintaining homogeneous coordinate system
    H = H / H[2, 2]
    # Inverse homography, from the next image frame to the base image frame
    H_inv = linalg.inv(H)
    H_inv = np.matmul(H_trans, H_inv)

    # Define the dimension of the full image frame, this value is constant.
    img_w = int(full_img.shape[1])
    img_h = int(full_img.shape[0])

    # Compute the hull containing the new image in the full image frame
    min_x, min_y, max_x, max_y = find_dimensions(next_img, H_inv)
    # Find and save the center point of the new image in the vector "vector"
    next_img_center_point = np.identity(3, np.float32)
    next_img_center_point[0, 2] = (next_img.shape[1] / 2)
    next_img_center_point[1, 2] = (next_img.shape[0] / 2)
    # Translate the center point in the full image frame
    next_img_center_point = np.matmul(H_inv, next_img_center_point)
    pts = np.array(next_img_center_point)
    vector.append(pts)
    # Add the offset value to the image.
    edge_1 = np.array([[int(min_x) - offset_value, int(min_y) - offset_value]])
    edge_3 = np.array([[int(max_x) + offset_value, int(max_y) + offset_value]])

    # The new image is "warped" in the full image frame
    next_img_warp = cv2.warpPerspective(next_img, H_inv, (img_w, img_h))
    # Create a black image with the size of the full image.
    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
    # Insert a white background over the just generated full frame in the zone caved by the new image
    (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    # Add the full image and the enlarged base image only where there are balck pixel form the enlarged_base_image,
    # this will leave a "hole" in which the new image is going to be stitched
    enlarged_full_img = cv2.add(enlarged_base_img, full_img, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
    # Add the new image to the just defined one in order to complete stitching
    final_img = cv2.add(enlarged_full_img, next_img_warp, dtype=cv2.CV_8U)
    return final_img, next_img_warp, vector, edge_1, edge_3


if __name__ == "__main__":
    # Initialize the csv to analyze the performance of the stitching algorithm
    with open('csv_plots.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SURF_kp full_img", "SURF_kp next_img", "number of good matches"])
    # Create the image list
    images = load_images_from_folder("sample_folder")
    base_img = images[0]
    # Initialize a frame for the full image. The frame need to contain the full final image
    huge_image = np.zeros((20000, 20000, 3), np.uint8)
    # insert the first image in the center of the created frame
    huge_image[
    int(huge_image.shape[0] / 2 - base_img.shape[0] / 2): int(huge_image.shape[0] / 2 - base_img.shape[0] / 2) +
                                                          base_img.shape[0],
    int(huge_image.shape[1] / 2 - base_img.shape[1] / 2): int(huge_image.shape[1] / 2 - base_img.shape[1] / 2) +
                                                          base_img.shape[1]] = base_img
    # Base image become the base image in the full frame
    base_img = huge_image
    # Insert the center point into the vector "vector" which contains the center point of all the images
    base_img_center_point = np.identity(3, np.float32)
    base_img_center_point[0, 2] = huge_image.shape[1] / 2
    base_img_center_point[1, 2] = huge_image.shape[0] / 2
    base_pts = np.array(base_img_center_point)
    # The offset value defines the number of pixel that need to be added to each side of the new image. This allows
    # to consider more area, and have a better stitching result by looking around the image frame and not only at just
    # the last image
    offset_value = 200
    vector = [base_pts]

    for i in range(1, len(images)):
        # Time is used for analysis test of the algorithm
        start_time = time.time()
        next_img = images[i]
        if i == 1:
            # Save image 1st and 3rd edges, since a rectangular shape is fully defined by just these 2 corners.
            edge_1 = np.array([[int(huge_image.shape[1] / 2 - images[0].shape[1] / 2) - offset_value,
                                int(huge_image.shape[0] / 2 - images[0].shape[0] / 2) - offset_value]])
            edge_3 = np.array([[int(huge_image.shape[1] / 2 + images[0].shape[1] / 2) + offset_value,
                                int(huge_image.shape[0] / 2 + images[0].shape[0] / 2) + offset_value]])
            # Crop from the full frame just the region of interest (last image) and compute key points and descriptor
            cropped_base = base_img[edge_1[0][1]:edge_3[0][1], edge_1[0][0]:edge_3[0][0], ...]
            img1_GS = cv2.GaussianBlur(cv2.cvtColor(cropped_base, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            kp_base_img, kp_descriptor_base_img = features_detection(img1_GS)
            kp_next_img, kp_descriptor_next_img = features_detection(img2_GS)
        else:
            cropped_base = final_img[edge_1[0][1]:edge_3[0][1], edge_1[0][0]:edge_3[0][0], ...]
            img1_GS = cv2.GaussianBlur(cv2.cvtColor(cropped_base, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            kp_base_img, kp_descriptor_base_img = features_detection(img1_GS)
            kp_next_img, kp_descriptor_next_img = features_detection(img2_GS)

        sel_matches = feature_matching(kp_descriptor_base_img, kp_descriptor_next_img)
        H = find_homography(kp_base_img, kp_next_img, sel_matches)
        H_trans = np.identity(3)
        # The H_trasn is a traslation related to the applyed cropping introduced before
        H_trans[0, 2] = edge_1[0][0]
        H_trans[1, 2] = edge_1[0][1]

        if i == 1:
            final_img, neg, next_center, edge_1, edge_3 = stitching(base_img, next_img, H, H_trans, vector,
                                                                    offset_value)
        else:
            final_img, neg, next_center, edge_1, edge_3 = stitching(final_img, next_img, H, H_trans, vector,
                                                                    offset_value)

        base_img = final_img
        with open('csv_plots.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([len(kp_base_img), len(kp_next_img), len(sel_matches), (time.time() - start_time)])
        print("iteration number ", i, " completed")

    final_img = cv2.medianBlur(final_img, 3)
    # PLOT THE CENTER ONTO THE IMAGE
    # for i in next_center:
    #     cv2.circle(final_img, (int(i[0, 2]), int(i[1, 2])), 5, (255, 255, 0), -1)
    # row_of_interest = []
    # with open('gps_xyz.csv', 'r') as file:
    #     csv_reader = csv.reader(file, delimiter=',')
    #     header = next(csv_reader)
    #     for row in csv_reader:
    #         row_of_interest.append(row)
    # for i in range(0, len(next_center)):
    #     if i % 5 == 0:
    #         # print(row_of_interest[i])
    #         cv2.putText(final_img, "X: " + str(row_of_interest[i][1]) + " Y: " + str(row_of_interest[i][2]),
    #                     (int(next_center[i][0][2] + 10), int(next_center[i][1][2])), cv2.FONT_ITALIC, 0.5,
    #                     (255, 255, 0))
    # cv2.imshow("next", final_img)

    # Save the image
    save_file_name = "img_save/cropping/crop" + str(images[0].shape[0]) + "X" + str(
        images[0].shape[1]) + "_N_img" + str(len(images)) + "_Offset_" + str(offset_value) + ".png"
    cv2.imwrite(save_file_name, final_img)
