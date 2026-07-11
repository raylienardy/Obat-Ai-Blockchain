"""
Validasi gambar apakah termasuk ciri kemasan obat.
"""
import cv2
import numpy as np


def is_medicine_package(image, strict=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    white_ratio = np.sum(mask_white > 0) / mask_white.size
    if not strict:
        return white_ratio >= 0.02
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_blobs = sum(1 for cnt in contours if 50 < cv2.contourArea(cnt) < 5000)
    mask_blue = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    mask_green = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    nature_ratio = (np.sum(mask_blue > 0) + np.sum(mask_green > 0)) / mask_blue.size
    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask_yellow = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    mask_bright_blue = cv2.inRange(hsv, (100, 100, 100), (120, 255, 255))
    bright_ratio = (np.sum(mask_red > 0) + np.sum(mask_yellow > 0) + np.sum(mask_bright_blue > 0)) / mask_red.size
    has_white = white_ratio >= 0.03
    has_text = text_blobs >= 3
    not_nature = nature_ratio < 0.4
    not_bright = bright_ratio < 0.1
    return has_white and has_text and not_nature and not_bright