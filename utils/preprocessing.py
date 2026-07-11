"""
Preprocessing gambar: deteksi bounding box, crop, enhance, dan persiapan dataset bersih.
"""
import cv2
import numpy as np
import os
from tqdm import tqdm
from .config import DATASET_DIR, DATASET_CLEAN_DIR, IMG_HEIGHT, IMG_WIDTH


def find_medicine_package(image):
    """Deteksi bounding box kemasan obat menggunakan OpenCV (legacy)."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    min_area = (h * w) * 0.02

    # Canny
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

    # Adaptive threshold
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

    # Otsu
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


def crop_and_enhance(image, rect):
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
    enhanced = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
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


def prepare_clean_dataset():
    dataset_asli_raw = DATASET_DIR / "asli"
    dataset_palsu_raw = DATASET_DIR / "palsu"
    os.makedirs(DATASET_CLEAN_DIR / "asli", exist_ok=True)
    os.makedirs(DATASET_CLEAN_DIR / "palsu", exist_ok=True)

    total = 0
    failed = 0

    for folder, raw_dir, clean_dir in [
        ('asli', dataset_asli_raw, DATASET_CLEAN_DIR / "asli"),
        ('palsu', dataset_palsu_raw, DATASET_CLEAN_DIR / "palsu")
    ]:
        if not raw_dir.exists():
            print(f"Folder {raw_dir} tidak ditemukan, melewati.")
            continue
        print(f"MEMPROSES FOLDER {folder.upper()}")
        for fname in tqdm(os.listdir(raw_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(raw_dir, fname))
                if img is None:
                    failed += 1
                    continue
                rect = find_medicine_package(img)
                if rect is None:
                    failed += 1
                enhanced = crop_and_enhance(img, rect)
                cv2.imwrite(os.path.join(clean_dir, fname), enhanced)
                total += 1
    print(f"Total diproses: {total}")
    print(f"Gagal deteksi (tapi tetap diproses): {failed}")
    return total, failed, []