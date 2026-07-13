# BASELINE PIPELINE — BlockchainAI

**Tanggal:** 13 Juli 2026
**Commit:** [hash commit saat ini]

## Pipeline

Input → detect_package() → crop_and_enhance() → MobileNetV2 → validation → Output

## Parameter Utama

| Parameter | Nilai |
|-----------|-------|
| YOLO confidence threshold | 0.5 |
| Classifier threshold | 0.5 |
| Confidence floor | 0.65 |
| White ratio min | 0.03 |
| Text blobs min | 5 |
| TTA | Nonaktif |
| Temperature | 1.0 |
| Debug mode | Nonaktif |
| Dataset uji | data/demo_test (15 gambar) |

## Model

- YOLO: runs/panadol_detector/weights/best.pt
- Classifier: models/panadol_v4_clean.keras