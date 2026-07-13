# BASELINE PIPELINE — BlockchainAI

**Baseline ID:** BL-001
**Tanggal Pembuatan:** YYYY-MM-DD
**Dibuat Oleh:** [Nama Engineer]
**Status:** ✅ AKTIF (tidak boleh diubah)

## Ringkasan

| Komponen | Versi/File |
|----------|------------|
| Detector | YOLO11 Nano |
| Model Detector | runs/panadol_detector/weights/best.pt |
| Classifier | MobileNetV2 |
| Model Classifier | models/panadol_v4_clean.keras |
| Dataset Uji | data/demo_test (15 gambar) |
| Konfigurasi | Tersimpan di config_snapshot.json |

## Metrik Utama

| Metrik | Nilai |
|--------|-------|
| Accuracy | 80% (12/15) |
| ASLI benar | 5/5 |
| PALSU benar | 0/5 |
| RANDOM ditolak | 2/5 |
| False Positive | 3 |
| False Negative | 5 |

## File Baseline

- [metrics.json](metrics.json)
- [model_info.json](model_info.json)
- [dataset_info.json](dataset_info.json)
- [config_snapshot.json](config_snapshot.json)
- [environment.json](environment.json)
- [rules.md](rules.md)

## Aturan Penting

- Baseline ini TIDAK BOLEH DIUBAH.
- Semua eksperimen harus dibandingkan dengan baseline ini.
- Jika model di-retrain, buat baseline baru dengan ID berbeda.