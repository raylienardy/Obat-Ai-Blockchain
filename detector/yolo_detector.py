"""
YOLO detector untuk mendeteksi kemasan obat.

File ini merupakan pengganti fungsi OpenCV:
find_medicine_package()

Output tetap sama:
(x, y, w, h) atau None
sehingga notebook lama tidak perlu banyak diubah.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import logging

from .config import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    DEVICE,
)

logger = logging.getLogger(__name__)

# ==========================================================
# Global Model Cache
# ==========================================================

_detector: Optional[YOLO] = None


# ==========================================================
# Load Model
# ==========================================================

def load_model() -> YOLO:
    """
    Memuat model YOLO sekali saja (lazy loading).
    """

    global _detector

    if _detector is None:

        model_path = Path(YOLO_MODEL)

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model tidak ditemukan:\n{model_path}"
            )

        logger.info("Loading YOLO model...")
        logger.info("Model : %s", model_path)
        logger.info("Device: %s", DEVICE)

        _detector = YOLO(str(model_path))

        logger.info("YOLO model loaded successfully.")

    return _detector


# ==========================================================
# Main Detection Function
# ==========================================================

def find_medicine_package_yolo(
    image: np.ndarray,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Drop-in replacement untuk find_medicine_package().

    Parameters
    ----------
    image : np.ndarray
        Image BGR hasil cv2.imread()

    confidence_threshold : float
        Confidence minimum

    Returns
    -------
    (x, y, w, h)
        Bounding box

    atau

    None
    """

    model = load_model()

    results = model(
        image,
        device=DEVICE,
        verbose=False,
    )

    best_bbox = None
    best_conf = 0.0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(box.conf.item())

            if conf < confidence_threshold:
                continue

            if conf > best_conf:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                best_conf = conf

                best_bbox = (
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                )

    if best_bbox is None:

        logger.info(
            "No medicine_package detected "
            "(threshold=%.2f)",
            confidence_threshold,
        )

        return None

    logger.info(
        "Detected medicine_package | Confidence=%.3f | BBox=%s",
        best_conf,
        best_bbox,
    )

    return best_bbox


# ==========================================================
# Utility
# ==========================================================

def detect_and_crop(
    image_path: str,
) -> Optional[np.ndarray]:
    """
    Membaca gambar, mendeteksi kemasan obat,
    kemudian mengembalikan hasil crop.
    """

    image = cv2.imread(image_path)

    if image is None:
        logger.error("Tidak dapat membaca gambar: %s", image_path)
        return None

    bbox = find_medicine_package_yolo(image)

    if bbox is None:
        return None

    x, y, w, h = bbox

    return image[y:y + h, x:x + w]