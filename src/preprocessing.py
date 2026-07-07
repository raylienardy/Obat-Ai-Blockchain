import cv2
import numpy as np

def find_medicine_package(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    min_area = (h * w) * 0.02

    # Strategi 1: Canny
    edges = cv2.Canny(blurred, 20, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=4)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and area > min_area:
            max_area = area
            best_rect = cv2.boundingRect(cnt)

    if best_rect is not None:
        x, y, bw, bh = best_rect
        if (bw * bh) < (h * w) * 0.85:
            return best_rect

    # Strategi 2: Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect2 = None
    max_area2 = 0
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area > max_area2 and area > min_area:
            max_area2 = area
            best_rect2 = cv2.boundingRect(cnt)

    if best_rect2 is not None:
        x, y, bw, bh = best_rect2
        if (bw * bh) < (h * w) * 0.85:
            return best_rect2

    # Strategi 3: Otsu
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel3)
    contours3, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect3 = None
    max_area3 = 0
    for cnt in contours3:
        area = cv2.contourArea(cnt)
        if area > max_area3 and area > min_area:
            max_area3 = area
            best_rect3 = cv2.boundingRect(cnt)

    if best_rect3 is not None:
        x, y, bw, bh = best_rect3
        if (bw * bh) < (h * w) * 0.85:
            return best_rect3

    return None


def crop_and_enhance(image, rect, img_height=224, img_width=224):
    if rect is None:
        h, w = image.shape[:2]
        rect = (0, 0, w, h)

    x, y, w_rect, h_rect = rect

    pad_x = int(w_rect * 0.02)
    pad_y = int(h_rect * 0.02)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w_rect + pad_x)
    y2 = min(image.shape[0], y + h_rect + pad_y)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = image

    enhanced = cv2.resize(cropped, (img_width, img_height))

    kernel_sharpen = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def is_medicine_package(image, strict=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Area putih / krem
    mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    white_ratio = np.sum(mask_white > 0) / mask_white.size

    if not strict:
        return white_ratio >= 0.02

    # 2. Blob teks/logo
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_blobs = sum(1 for cnt in contours if 50 < cv2.contourArea(cnt) < 5000)

    # 3. Filter warna alam
    mask_blue = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    mask_green = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    nature_ratio = (np.sum(mask_blue > 0) + np.sum(mask_green > 0)) / mask_blue.size

    # 4. Filter warna mencolok (merah, kuning, biru terang)
    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask_yellow = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    mask_bright_blue = cv2.inRange(hsv, (100, 100, 100), (120, 255, 255))
    bright_ratio = (np.sum(mask_red > 0) + np.sum(mask_yellow > 0) + np.sum(mask_bright_blue > 0)) / mask_red.size

    has_white = white_ratio >= 0.03
    has_text = text_blobs >= 3
    not_nature = nature_ratio < 0.4
    not_bright = bright_ratio < 0.1

    return has_white and has_text and not_nature and not_bright