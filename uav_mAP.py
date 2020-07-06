import cv2
import numpy as np


def features_detection(img, type):
    if type == "SURF":
        descriptor = cv2.xfeatures2d.SURF_create()
        # Other than none we can pass a mask to cover parts of the image that is not needed
        key_pts, features = descriptor.detectAndCompute(img, None)
    elif type == "ORB":
        descriptor = cv2.ORB_create(nfeatures=10000)
        # Other than none we can pass a mask to cover parts of the image that is not needed
        kp_pt, kp_descriptor = descriptor.detectAndCompute(img, None)
    return kp_pt, kp_descriptor


def feature_matching(img_descriptor, query_descriptor):
    # Create a Brute Force matcher object, for ORB is advised to use HAMMING distance
    # crossCehck=True sotre only cross correlated matches between the two images

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match the descriptors giving as input the two images descriptor

    matches = bf.match(img_descriptor, query_descriptor)
    dmatches = sorted(matches, key=lambda x: x.distance)
    return dmatches

def find_homography(kp_pt_img, kp_pt_query, dmatches):
    src_pts = np.float32([kp_pt_img[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_pt_query[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)

    ## find homography matrix and do perspective transform
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return mask


img1 = cv2.imread("img_folder/WhatsApp Image 2020-07-06 at 11.52.39.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("img_folder/WhatsApp Image 2020-07-06 at 11.52.40.jpeg", cv2.IMREAD_GRAYSCALE)


kp_img, kp_descriptor_img = features_detection(img1, "ORB")
kp_query, kp_descriptor_query = features_detection(img1, "ORB")

dmatches = feature_matching(kp_descriptor_img, kp_descriptor_query)
print(dmatches)
mask = find_homography(kp_img, kp_query, dmatches)
print(mask)


#img_mod = cv2.drawKeypoints(img1, kp, None)
#cv2.imshow("keypoint", img_mod)
#print(p)
#cv2.waitKey()
