"""
Fungsi inferensi dan loading model classifier.
"""
import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from .config import MODEL_DIR, MODEL_FILENAME
from .preprocessing import find_medicine_package, crop_and_enhance


def load_verify_model():
    """Memuat model classifier dari disk."""
    model_path = MODEL_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    return tf.keras.models.load_model(model_path)


def predict_medicine(image_path, verify_model, threshold=0.5, confidence_floor=0.80,
                     white_ratio_min=0.03, min_text_blobs=3):
    if verify_model is None:
        return "ERROR", "Model belum dimuat", 0.0, None
    img = cv2.imread(image_path)
    if img is None:
        return "ERROR", "Gambar tidak ditemukan", 0.0, None
    original = img.copy()
    rect = find_medicine_package(img)
    if rect is None:
        rect = (0, 0, img.shape[1], img.shape[0])
    enhanced = crop_and_enhance(img, rect)
    if enhanced is None:
        return "ERROR", "Gagal memproses gambar", 0.0, original
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0) / 255.0
    pred = verify_model.predict(img_array, verbose=0)[0][0]
    if pred >= threshold:
        label = "❌ PALSU"
        confidence = pred
    else:
        label = "✅ ASLI"
        confidence = 1.0 - pred
    if confidence < confidence_floor:
        return "DITOLAK", "❌ Objek tidak dikenal / bukan obat", confidence, enhanced
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    white_ratio = np.sum(mask_white > 0) / mask_white.size
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_blobs = sum(1 for cnt in contours if 50 < cv2.contourArea(cnt) < 5000)
    has_white = white_ratio >= white_ratio_min
    has_text = text_blobs >= min_text_blobs
    if not (has_white or has_text):
        return "DITOLAK", "❌ Objek bukan obat / tidak ada ciri kemasan", confidence, enhanced
    return "TERVERIFIKASI", label, confidence, enhanced


def predict_and_display(image_path, verify_model, threshold=0.5):
    status, label, confidence, enhanced = predict_medicine(image_path, verify_model, threshold)
    original = cv2.imread(image_path)
    if original is not None:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title("Gambar Asli")
        plt.axis('off')
        if enhanced is not None:
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            plt.subplot(1, 2, 2)
            plt.imshow(enhanced_rgb)
            plt.title("Hasil Crop & Enhance")
            plt.axis('off')
        plt.suptitle(f"{label}\nConfidence: {confidence*100:.2f}%", fontsize=14)
        plt.show()
    print("="*60)
    print(f"File    : {os.path.basename(image_path)}")
    print(f"Status  : {status}")
    print(f"Hasil   : {label}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*60)
    return status, label, confidence
  
def run_batch_test(demo_dir, verify_model, threshold=0.5):
    """
    Melakukan inferensi terhadap seluruh gambar di folder demo_test.
    Menampilkan hasil dan mengembalikan dictionary ringkasan.
    """
    demo_dir = Path(demo_dir)
    if not demo_dir.exists():
        print("❌ Folder demo_test tidak ditemukan.")
        return None

    test_files = sorted([f for f in os.listdir(demo_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not test_files:
        print("⚠️ Folder demo_test kosong.")
        return None

    results = {"ASLI": 0, "PALSU": 0, "DITOLAK": 0, "ERROR": 0}
    print("="*70)
    print(f"BATCH TESTING PADA FOLDER: {demo_dir}")
    print("="*70)

    for fname in test_files:
        path = str(demo_dir / fname)
        status, label, confidence, _ = predict_medicine(path, verify_model, threshold)
        if status == "DITOLAK":
            results["DITOLAK"] += 1
            print(f"{fname:<40} {label}")
        elif status == "ERROR":
            results["ERROR"] += 1
            print(f"{fname:<40} {label}")
        else:
            symbol = "✅" if "ASLI" in label else "❌"
            print(f"{fname:<40} {symbol} {label}  (confidence: {confidence*100:.1f}%)")
            if "ASLI" in label:
                results["ASLI"] += 1
            else:
                results["PALSU"] += 1

    print("\n" + "="*70)
    print("RINGKASAN")
    print("="*70)
    print(f"✅ ASLI    : {results['ASLI']}")
    print(f"❌ PALSU   : {results['PALSU']}")
    print(f"🚫 DITOLAK : {results['DITOLAK']} (bukan obat / tidak terdeteksi)")
    print(f"⚠️  ERROR  : {results['ERROR']}")
    return results