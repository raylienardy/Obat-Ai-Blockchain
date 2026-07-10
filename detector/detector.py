"""
Modul deteksi kemasan obat menggunakan YOLO11 Nano.
Menyediakan pengganti drop-in untuk find_medicine_package() lama.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple
import logging

from detector.config import YOLO_MODEL, CONFIDENCE_THRESHOLD, DEVICE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cache model
_detector: Optional[YOLO] = None


def load_model() -> YOLO:
    """
    Memuat model YOLO dari path yang ditentukan di config.
    Model disimpan di variabel global untuk efisiensi.
    """
    global _detector
    if _detector is None:
        logger.info("Loading YOLO model from %s", YOLO_MODEL)
        if not YOLO_MODEL.exists():
            raise FileNotFoundError(f"Model YOLO tidak ditemukan: {YOLO_MODEL}")
        _detector = YOLO(str(YOLO_MODEL))
        # Pindahkan ke GPU jika tersedia
        _detector.to(device=DEVICE)
        logger.info("YOLO model loaded successfully.")
    return _detector


def find_medicine_package_yolo(image: np.ndarray,
                               confidence_threshold: float = CONFIDENCE_THRESHOLD
                               ) -> Optional[Tuple[int, int, int, int]]:
    """
    Deteksi kemasan obat menggunakan YOLO.
    Menggantikan find_medicine_package() yang lama.

    Args:
        image: Gambar input (numpy array BGR, hasil cv2.imread).
        confidence_threshold: Ambang batas keyakinan minimum.

    Returns:
        Tuple (x, y, w, h) dari bounding box dengan confidence tertinggi,
        atau None jika tidak ada deteksi.
    """
    model = load_model()
    results = model(image, verbose=False)

    best_conf = 0.0
    best_bbox = None

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                best_bbox = (int(x1), int(y1), int(w), int(h))

    if best_bbox is not None:
        logger.info("Detected medicine_package | Confidence: %.3f | Bbox: %s",
                     best_conf, best_bbox)
    else:
        logger.info("No medicine_package detected (threshold=%.2f)",
                     confidence_threshold)

    return best_bbox


def detect_and_crop(image_path: str) -> Optional[np.ndarray]:
    """
    Membaca gambar, mendeteksi bounding box, lalu mengembalikan area yang di-crop.

    Args:
        image_path: Path ke file gambar.

    Returns:
        Cropped image (numpy array BGR) atau None jika gagal/tidak terdeteksi.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Gagal membaca gambar: %s", image_path)
        return None

    bbox = find_medicine_package_yolo(img)
    if bbox is None:
        logger.warning("Tidak ada objek terdeteksi di %s", image_path)
        return None

    x, y, w, h = bbox
    cropped = img[y:y+h, x:x+w]
    return cropped