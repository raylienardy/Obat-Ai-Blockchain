"""
Konfigurasi untuk detektor YOLO.
"""

from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent

YOLO_MODEL = ROOT / "runs" / "panadol_detector" / "weights" / "best.pt"

CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 640

# otomatis pilih GPU bila ada
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"