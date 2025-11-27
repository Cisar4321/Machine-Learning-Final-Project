import cv2
import numpy as np
from pathlib import Path

def segment_primary_leaf(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_dark_green = np.array([20, 40, 20])
    upper_dark_green = np.array([80, 255, 255])
    lower_light_green = np.array([30, 10, 40])
    upper_light_green = np.array([90, 255, 255])
    lower_yellow = np.array([14, 30, 80])
    upper_yellow = np.array([35, 255, 255])
    m1 = cv2.inRange(hsv, lower_dark_green, upper_dark_green)
    m2 = cv2.inRange(hsv, lower_light_green, upper_light_green)
    m3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.medianBlur(mask, 7)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        primary_mask = mask
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        primary_mask = np.where(labels == idx, 255, 0).astype(np.uint8)
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=primary_mask)
    overlay = image_bgr.copy()
    overlay[primary_mask == 0] = (overlay[primary_mask == 0] * 0.3).astype(np.uint8)
    return segmented, primary_mask, overlay

def save_outputs(segmented, mask, overlay, out_dir, base_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{base_name}_primary.png"), segmented)
    cv2.imwrite(str(out_dir / f"{base_name}_mask.png"), mask)
    cv2.imwrite(str(out_dir / f"{base_name}_overlay.png"), overlay)
