import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image


# ------------------------------
# Unified helpers 
# ------------------------------

def delineate_damage(mask, reference_image, min_area=500):
    """
    Takes a binary mask and draws delineated contours on a copy of reference_image.
    Used by all detection functions to avoid duplication.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    delineated = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:  # ignore small noise
            cv2.drawContours(delineated, [cnt], -1, (0, 0, 255), 2)  # red outline

    return delineated


def visualize_pipeline(pre, post, diff, mask, delineated):
    """
    Unified visualization: Pre, Post, Diff, Mask, Post with delineation.
    """
    cv2.imshow("Pre", pre)
    cv2.imshow("Post", post)
    cv2.imshow("Diff Map", diff)
    cv2.imshow("Damage Mask", mask)
    cv2.imshow("Delineated Damage Areas", delineated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------------
# Alignment Functions
# ------------------------------

def align_images(img1, img2):
    """
    Rough alignment using ORB feature matching + homography.
    Works best when both images cover the same scene at slightly different angles.
    """
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, m in enumerate(matches):
        points1[i, :] = kp1[m.queryIdx].pt
        points2[i, :] = kp2[m.trainIdx].pt

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


# ------------------------------
# change_detection functions
# ------------------------------

def change_detection(pre_path, post_path, visualize=True):
    pre = cv2.imread(pre_path)
    post = cv2.imread(post_path)

    h, w = pre.shape[:2]
    post = cv2.resize(post, (w, h))

    post_clean = post
    aligned_post = align_images(pre, post_clean)

    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(aligned_post, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(pre_gray, post_gray, full=True)
    diff = (1 - diff) * 255
    diff = diff.astype(np.uint8)

    _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    delineated = delineate_damage(mask, post_gray)

    if visualize:
        visualize_pipeline(pre, post, diff, mask, delineated)

    return mask


# ------------------------------
# simple_change_detection
# ------------------------------

def simple_change_detection(pre_path, post_path, visualize=True):
    pre = cv2.imread(pre_path)
    post = cv2.imread(post_path)

    h, w = pre.shape[:2]
    post = cv2.resize(post, (w, h))

    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    pre_blur = cv2.GaussianBlur(pre_gray, (5, 5), 0)
    post_blur = cv2.GaussianBlur(post_gray, (5, 5), 0)

    diff = cv2.absdiff(pre_blur, post_blur)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    _, mask = cv2.threshold(diff_norm, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    delineated = delineate_damage(mask, post_gray)

    if visualize:
        visualize_pipeline(pre, post, diff_norm, mask, delineated)

    return mask


# ------------------------------
# block_change_detection
# ------------------------------

def block_change_detection(pre_path, post_path, block_size=8, visualize=True):
    # Load and resize
    pre = cv2.imread(pre_path)
    post = cv2.imread(post_path)

    h, w = pre.shape[:2]
    post = cv2.resize(post, (w, h))

    # Convert to grayscale
    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    # Optional: histogram match normalization
    pre_gray = cv2.equalizeHist(pre_gray)
    post_gray = cv2.equalizeHist(post_gray)

    # Optional: Gaussian blur to reduce minor noise
    pre_blur = cv2.GaussianBlur(pre_gray, (5, 5), 0)
    post_blur = cv2.GaussianBlur(post_gray, (5, 5), 0)

    diff_map = np.zeros_like(pre_gray, dtype=np.uint8)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)

            pre_block = pre_blur[y:y_end, x:x_end]
            post_block = post_blur[y:y_end, x:x_end]

            diff_val = abs(float(pre_block.mean()) - float(post_block.mean()))
            diff_map[y:y_end, x:x_end] = np.clip(diff_val * 5, 0, 255)

    _, mask = cv2.threshold(diff_map, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    delineated = delineate_damage(mask, post_gray)

    if visualize:
        visualize_pipeline(pre, post, diff_map, mask, delineated)

    return mask