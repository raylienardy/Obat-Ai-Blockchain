"""
Konfigurasi untuk detektor YOLO.
"""
from pathlib import Path

# Root proyek (detector/ ada di dalam root)
ROOT = Path(__file__).parent.parent

# Path model YOLO terlatih
YOLO_MODEL = ROOT / "runs" / "panadol_detector" / "weights" / "best.pt"

# Parameter deteksi
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 640
DEVICE = 0  # GPU