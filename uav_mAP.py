import cv2
import numpy as np
import utils
import math

from numpy import linalg

def findDimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

        if (max_x == None or normal_pt[0, 0] > max_x):
            max_x = normal_pt[0, 0]

        if (max_y == None or normal_pt[1, 0] > max_y):
            max_y = normal_pt[1, 0]

        if (min_x == None or normal_pt[0, 0] < min_x):
            min_x = normal_pt[0, 0]

        if (min_y == None or normal_pt[1, 0] < min_y):
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)


def features_detection(img, type):
    if type == "ORB":
        descriptor = cv2.ORB_create(nfeatures=1000)
        # Other than none we can pass a mask to cover parts of the image that is not needed
        kp_pt, kp_descriptor = descriptor.detectAndCompute(img, None)
        return kp_pt, kp_descriptor


def feature_matching(base_img_descriptor, next_img_descriptor):
    # Create a Brute Force matcher object, for ORB is advised to use HAMMING distance
    # crossCheck=True store only cross correlated matches between the two images

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match the descriptors giving as input the two images descriptor

    matches = bf.match(base_img_descriptor, next_img_descriptor)
    # Sort the matches in order of distance

    sorted_matches = sorted(matches, key=lambda m: m.distance)
    return sorted_matches


def find_homography(kp_pt_img, kp_pt_query, sorted_matches):
    src_pts = np.float32([kp_pt_img[m.queryIdx].pt for m in sorted_matches[0:40]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_pt_query[m.trainIdx].pt for m in sorted_matches[0:40]]).reshape(-1, 1, 2)
    ## find homography matrix and do perspective transform by using of RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("No Homography")
    else:
        print(H)
        return H


def stitching(base_img, next_img, H):
    H = H / H[2, 2]
    H_inv = linalg.inv(H)

    (min_x, min_y, max_x, max_y) = findDimensions(next_img, H_inv)

    # Adjust max_x and max_y by base img size
    max_x = max(max_x, base_img.shape[1])
    max_y = max(max_y, base_img.shape[0])

    move_h = np.matrix(np.identity(3), np.float32)

    if (min_x < 0):
        move_h[0, 2] += -min_x
        max_x += -min_x

    if (min_y < 0):
        move_h[1, 2] += -min_y
        max_y += -min_y


    mod_inv_h = move_h * H_inv

    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))



    # Warp the new image given the homography from the old image
    base_img_warp = cv2.warpPerspective(next_img, move_h, (img_w, img_h))
    next_img_warp = cv2.warpPerspective(base_img, mod_inv_h, (img_w, img_h))

    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)


    # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
    # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

    # Create a mask from the warped image for constructing masked composite
    (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
                                    0, 255, cv2.THRESH_BINARY)

    enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                mask=np.bitwise_not(data_map),
                                dtype=cv2.CV_8U)

    # Now add the warped image
    final_img = cv2.add(enlarged_base_img, next_img_warp,
                        dtype=cv2.CV_8U)

    cv2.imshow("try", cv2.resize(final_img,(640,480)))
    cv2.waitKey()


if __name__ == "__main__":
    base_img = cv2.imread("img_folder/IMG_7104.jpg")
    next_img = cv2.imread("img_folder/IMG_7105.jpg")
    img1_GS = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    img2_GS = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    kp_base_img, kp_descriptor_base_img = features_detection(img2_GS, "ORB")
    kp_next_img, kp_descriptor_next_img = features_detection(img1_GS, "ORB")

    sorted_matches = feature_matching(kp_descriptor_base_img, kp_descriptor_next_img)

    H = find_homography(kp_base_img, kp_next_img, sorted_matches)

    stitching(base_img, next_img, H)
