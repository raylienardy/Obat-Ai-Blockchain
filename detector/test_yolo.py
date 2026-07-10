"""
Script pengujian YOLO pada satu gambar.
Menampilkan confidence, bounding box, dan menyimpan anotasi.
"""
import argparse
from pathlib import Path
import cv2
import logging
from detector.detector import load_model, find_medicine_package_yolo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Uji deteksi YOLO pada satu gambar.")
    parser.add_argument("image", type=str, help="Path gambar input")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("File tidak ditemukan: %s", image_path)
        return

    img = cv2.imread(str(image_path))
    if img is None:
        logger.error("Gagal membaca gambar.")
        return

    model = load_model()
    results = model(img, verbose=False)

    # Simpan anotasi
    annotated = results[0].plot()
    out_path = OUTPUT_DIR / f"yolo_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)
    logger.info("Anotasi disimpan ke %s", out_path)

    # Tampilkan deteksi
    bbox = find_medicine_package_yolo(img, confidence_threshold=args.conf)
    if bbox:
        x, y, w, h = bbox
        logger.info("Deteksi: x=%d, y=%d, w=%d, h=%d", x, y, w, h)
    else:
        logger.info("Tidak ada deteksi.")


if __name__ == "__main__":
    main()