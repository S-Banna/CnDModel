import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def align_images(img1, img2):
    """
    Rough alignment using ORB feature matching + homography.
    Works best when both images cover the same scene at slightly different angles.
    """
    #ADDED
    pre_path = r"C:\Users\user\CnDModel\data\pairs\pre\33.8325 35.51375 zoomed pre image up.png"
    post_path = r"C:\Users\user\CnDModel\data\pairs\post\33.8325 35.51375 zoomed post image up.png"
    
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # match keypoints
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # use top matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, m in enumerate(matches):
        points1[i, :] = kp1[m.queryIdx].pt
        points2[i, :] = kp2[m.trainIdx].pt

    # estimate homography
    h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width = img1.shape[:2]
    aligned = cv2.warpPerspective(img2, h, (width, height))

    return aligned

def remove_blue_marks(image):
    """
    Removes blue annotations by detecting the blue hue range.
    Converts detected blue pixels to neutral grey to avoid biasing change detection.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    image[mask > 0] = [128, 128, 128]
    return image

def change_detection(pre_path, post_path, visualize=True):
    pre = cv2.imread(pre_path)
    post = cv2.imread(post_path)

    # optional resize to same scale
    h, w = pre.shape[:2]
    post = cv2.resize(post, (w, h))

    # clean up blue rectangles
    # changed post_clean = remove_blue_marks(post) because the pictures do not have blue circles anymore
    post_clean = post
    # align roughly
    aligned_post = align_images(pre, post_clean)

    # convert both to grayscale
    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(aligned_post, cv2.COLOR_BGR2GRAY)

    # compute SSIM difference map
    score, diff = ssim(pre_gray, post_gray, full=True)
    diff = (1 - diff) * 255
    diff = diff.astype(np.uint8)

    # threshold and clean
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    if visualize:
        cv2.imshow("Diff Map", diff)
        cv2.imshow("Threshold", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return thresh
