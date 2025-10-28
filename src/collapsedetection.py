import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def align_images(img1, img2):
    """
    Roughly aligns two satellite images using ORB feature matching and homography.

    This helps correct for slight shifts or angle differences between
    the pre- and post-conflict images before performing change detection.
    """

    # Initialize ORB (Oriented FAST and Rotated BRIEF) feature detector
    orb = cv2.ORB_create(5000)

    # Detect keypoints and compute feature descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute-force match features using Hamming distance (works for binary descriptors like ORB)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    # Sort matches by distance (lower distance = better match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoint coordinates
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, m in enumerate(matches):
        points1[i, :] = kp1[m.queryIdx].pt
        points2[i, :] = kp2[m.trainIdx].pt

    # Estimate homography matrix using RANSAC to reject outliers
    h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp the post-image to align it with the pre-image
    height, width = img1.shape[:2]
    aligned = cv2.warpPerspective(img2, h, (width, height))

    return aligned


def remove_blue_marks(image):
    """
    Removes blue annotations often present on satellite images (e.g., Google Earth labels).
    Detects blue hues in HSV color space and replaces them with neutral gray.

    This prevents annotation marks from being mistaken as structural changes.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Replace blue pixels with gray (128,128,128)
    image[mask > 0] = [128, 128, 128]

    return image


def change_detection(pre_path, post_path, visualize=True):
    """
    Detects damaged or changed areas between two satellite images (pre- and post-conflict).

    Steps:
    1. Read and resize images to the same dimensions
    2. Remove blue labels or marks
    3. Align images using feature matching
    4. Compute Structural Similarity (SSIM) difference map
    5. Threshold the map to isolate major differences (potentially damaged areas)

    Args:
        pre_path (str): Path to the pre-conflict image
        post_path (str): Path to the post-conflict image
        visualize (bool): Whether to display difference maps

    Returns:
        np.ndarray: Binary mask highlighting changed/damaged regions
    """
    pre = cv2.imread(pre_path)
    post = cv2.imread(post_path)
    
    #Added if pre is None check — avoids crashes when an image path is wrong
    if pre is None or post is None:
        print("Error: One or both image paths are invalid.")
        return None

    # Ensure both images have same dimensions
    h, w = pre.shape[:2]
    post = cv2.resize(post, (w, h))

    # Remove Google Earth markings (if any)
    post_clean = remove_blue_marks(post)

    # Align post-image to pre-image
    aligned_post = align_images(pre, post_clean)

    # Convert to grayscale for SSIM comparison
    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(aligned_post, cv2.COLOR_BGR2GRAY)

    # Compute SSIM score and difference map
    score, diff = ssim(pre_gray, post_gray, full=True)
    diff = (1 - diff) * 255  # Convert similarity → difference
    diff = diff.astype(np.uint8)

    # Threshold the difference to create a binary mask
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    
    # Morphological opening removes small noise regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    if visualize:
        cv2.imshow("Difference Map", diff)
        cv2.imshow("Thresholded Damage Mask", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return thresh
