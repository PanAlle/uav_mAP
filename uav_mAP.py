import cv2
import numpy as np
import math
from numpy import linalg

def findDimensions(image, homography):
    # Initialize 1x3 array of 1 to hold x,y,channel
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    # Get image Height and Width
    (y, x) = image.shape[:2]

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


        # Transform next_image edges into the base space
        hp = np.dot(np.array(homography, np.float32), np.array(pt, np.float32).reshape(-1,1))

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


def features_detection(img, type):
    if type == "ORB":
        detected_points = 4000
        descriptor = cv2.ORB_create(nfeatures=detected_points)
    elif type == "SURF":
        descriptor = cv2.xfeatures2d.SURF_create()
    # Other than none we can pass a mask to cover parts of the image that is not needed
    kp_pt, kp_descriptor = descriptor.detectAndCompute(img, None)
    return kp_pt, kp_descriptor


def feature_matching(base_img_descriptor, next_img_descriptor):
    # Create a Brute Force matcher object, for ORB is advised to use HAMMING distance
    # crossCheck=True store only cross correlated matches between the two images

    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match the descriptors giving as input the two images descriptor

    #matches = bf.match(base_img_descriptor, next_img_descriptor)
    # Sort the matches in order of distance
    bf = cv2.BFMatcher()
    #give the two best matches for each descriptor
    matches = bf.knnMatch(base_img_descriptor, next_img_descriptor, k=2)
    #sorted_matches = sorted(matches, key=lambda m: m.distance)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good


def find_homography(kp_base_img, kp_next_img, sorted_matches):
    src_pts = np.float32([kp_base_img[m.queryIdx].pt for m in sorted_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_next_img[m.trainIdx].pt for m in sorted_matches]).reshape(-1, 1, 2)
    ## find homography matrix and do perspective transform by using of RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("No Homography")
    else:
        print(H)
        return H


def stitching(base_img, next_img, H):
    # Normalize to maintaing homogeneus coordianate system
    H = H / H[2, 2]
    #Inverse matrix
    H_inv = linalg.inv(H)
    print("Inverse matrix",H_inv)
    #Find the rectangular hull containing the next_img in the base_img reference frame
    (min_x, min_y, max_x, max_y) = findDimensions(next_img, H_inv)

    # Adjust max_x, max_y in order to include also the first image
    max_x = max(max_x, base_img.shape[1])
    max_y = max(max_y, base_img.shape[0])
    # Define the move vector for each pixel
    move_h = np.identity(3 , np.float32)
    # Translate the origin of the image in the positive part of the plane, in order to see it
    if (min_x < 0):
        move_h[0, 2] += -min_x
        max_x += -min_x

    if (min_y < 0):
        move_h[1, 2] += -min_y
        max_y += -min_y

    # Inverse translation vector, from H space to normal space
    mod_inv_h = np.dot(H_inv,move_h)
    # Return the closest integer near a given number
    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

    # Warp the new image given the homography from the old image
    base_img_warp = cv2.warpPerspective(base_img, move_h, (img_w, img_h))
    next_img_warp = cv2.warpPerspective(next_img, mod_inv_h, (img_w, img_h))

    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
    #enlarged_next_img = np.zeros((img_h, img_w, 3), np.uint8)

    print(int(mod_inv_h[1,2]))
    # Create a mask from the warped image for constructing masked composite (insert black base on next image, covering the first one)
    (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, mask=np.bitwise_not(data_map),dtype=cv2.CV_8U)
    #enlarged_next_img = cv2.add(enlarged_next_img, next_img_warp, mask=np.bitwise_not(data_map),dtype=cv2.CV_8U)
    # Now add the warped image with 8bit/pixel (0 - 255)
    final_img = cv2.add(enlarged_base_img, next_img_warp, dtype=cv2.CV_8U)
    #cv2.imshow("try", cv2.resize(final_img,(640,480)))
    #final_img[int(mod_inv_h[1, 2]):final_img.shape[0], int(mod_inv_h[0, 2]): final_img.shape[1]] = next_img
    cv2.imshow("try",cv2.medianBlur(final_img, 3))
    cv2.imwrite("img_folder/test4_orb3000_07dlim.jpeg", final_img)


    last_p1 = np.ones(3, np.float32)
    last_p2 = np.ones(3, np.float32)
    last_p3 = np.ones(3, np.float32)
    last_p4 = np.ones(3, np.float32)
    # Last image position
    (y, x) = next_img_warp.shape[:2]

    last_p1[:2] = [0 + mod_inv_h[0,2], 0 + mod_inv_h[1,2]]
    last_p2[:2] = [x + mod_inv_h[0,2], 0 + mod_inv_h[1,2]]
    last_p3[:2] = [0 + mod_inv_h[0,2], y + mod_inv_h[1,2]]
    last_p4[:2] = [x + mod_inv_h[0,2], y + mod_inv_h[1,2]]


    return final_img[int(mod_inv_h[1, 2]):next_img_warp.shape[0] + int(mod_inv_h[1, 2]), int(mod_inv_h[0, 2]): final_img.shape[1] + int(mod_inv_h[0, 2])], final_img



if __name__ == "__main__":
    base_img = cv2.imread("img_folder/WhatsApp Image 2020-07-08 at 17.08.02.jpeg")
    next_img = cv2.imread("img_folder/WhatsApp Image 2020-07-08 at 17.08.02(1).jpeg")

    # base_img = cv2.imread("img_folder/IMG_7105.jpg")
    # next_img = cv2.imread("img_folder/IMG_7106.jpg")

    #img1_GS = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    #img2_GS = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    img1_GS = cv2.GaussianBlur(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), (5,5), 0)
    img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY), (5,5), 0)

    kp_base_img, kp_descriptor_base_img = features_detection(img1_GS, "SURF")
    kp_next_img, kp_descriptor_next_img = features_detection(img2_GS, "SURF")

    sorted_matches = feature_matching(kp_descriptor_base_img, kp_descriptor_next_img)

    H = find_homography(kp_base_img, kp_next_img, sorted_matches)

    final_img_crp, final_img = stitching(base_img, next_img, H)
    # Second iteration

    base_img = final_img
    next_next_img = cv2.imread("img_folder/WhatsApp Image 2020-07-08 at 17.08.03.jpeg")

    img1_GS = cv2.GaussianBlur(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    kp_base_img, kp_descriptor_base_img = features_detection(img1_GS, "SURF")
    kp_next_img, kp_descriptor_next_img = features_detection(img2_GS, "SURF")

    sorted_matches = feature_matching(kp_descriptor_base_img, kp_descriptor_next_img)

    H = find_homography(kp_base_img, kp_next_img, sorted_matches)

    final_img_crp, final_img = stitching(base_img, next_next_img, H)

    # Third iteration

    base_img = final_img
    next_next_img = cv2.imread("img_folder/WhatsApp Image 2020-07-08 at 17.08.03(1).jpeg")

    img1_GS = cv2.GaussianBlur(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    kp_base_img, kp_descriptor_base_img = features_detection(img1_GS, "SURF")
    kp_next_img, kp_descriptor_next_img = features_detection(img2_GS, "SURF")

    sorted_matches = feature_matching(kp_descriptor_base_img, kp_descriptor_next_img)

    H = find_homography(kp_base_img, kp_next_img, sorted_matches)

    final_img_crp, final_img = stitching(base_img, next_next_img, H)

    # Fourth iteration

    base_img = final_img
    next_next_img = cv2.imread("img_folder/WhatsApp Image 2020-07-08 at 17.08.04.jpeg")

    img1_GS = cv2.GaussianBlur(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    img2_GS = cv2.GaussianBlur(cv2.cvtColor(next_next_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    kp_base_img, kp_descriptor_base_img = features_detection(img1_GS, "SURF")
    kp_next_img, kp_descriptor_next_img = features_detection(img2_GS, "SURF")

    sorted_matches = feature_matching(kp_descriptor_base_img, kp_descriptor_next_img)

    H = find_homography(kp_base_img, kp_next_img, sorted_matches)

    final_img_crp, final_img = stitching(base_img, next_next_img, H)

    cv2.imshow("next", cv2.resize(cv2.medianBlur(final_img, 3), (640,480)))
    cv2.waitKey()