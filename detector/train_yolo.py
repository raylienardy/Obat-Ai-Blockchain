"""
Script pelatihan YOLO11 Nano untuk mendeteksi medicine_package.
"""
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi
ROOT = Path(__file__).parent.parent
DATA_YAML = ROOT / "data" / "yolo_dataset" / "data.yaml"
MODEL_NAME = "yolo11n.pt"
EPOCHS = 150
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0
WORKERS = 8
PATIENCE = 20
SEED = 42


def main():
    logger.info("Mulai training YOLO...")

    if not DATA_YAML.exists():
        logger.error("Data YAML tidak ditemukan: %s", DATA_YAML)
        return

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        cache="disk",
        patience=PATIENCE,
        project="runs",
        name="panadol_detector",
        exist_ok=True,
        seed=SEED,
        pretrained=True,
        optimizer="auto",
        plots=True,
        save=True,
    )
    logger.info("Training selesai. Model disimpan di runs/panadol_detector/weights/best.pt")


if __name__ == "__main__":
    main()