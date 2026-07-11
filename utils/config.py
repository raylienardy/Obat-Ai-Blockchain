"""
Konfigurasi global proyek BlockchainAI.
"""
from pathlib import Path

# Root proyek (utils/ ada di dalam root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
DATASET_CLEAN_DIR = DATA_DIR / "dataset_clean"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Parameter gambar
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Nama file model
MODEL_FILENAME = "panadol_v4_clean.keras"
TFLITE_FILENAME = "panadol_v4_clean.tflite"