"""
Fungsi-fungsi preprocessing, training, dan inferensi.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
from pathlib import Path

from .config import (
    DATA_DIR,
    DATASET_DIR,
    DATASET_CLEAN_DIR,
    MODEL_DIR,
    IMG_HEIGHT,
    IMG_WIDTH
)

# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def find_medicine_package(image):
    """Deteksi bounding box kemasan obat menggunakan OpenCV (legacy)."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    min_area = (h * w) * 0.02

    # Canny
    edges = cv2.Canny(blurred, 20, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=4)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and area > min_area:
            max_area = area
            best_rect = cv2.boundingRect(cnt)
    if best_rect is not None:
        x, y, bw, bh = best_rect
        if (bw * bh) < (h * w) * 0.85:
            return best_rect

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect2 = None
    max_area2 = 0
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area > max_area2 and area > min_area:
            max_area2 = area
            best_rect2 = cv2.boundingRect(cnt)
    if best_rect2 is not None:
        x, y, bw, bh = best_rect2
        if (bw * bh) < (h * w) * 0.85:
            return best_rect2

    # Otsu
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel3)
    contours3, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect3 = None
    max_area3 = 0
    for cnt in contours3:
        area = cv2.contourArea(cnt)
        if area > max_area3 and area > min_area:
            max_area3 = area
            best_rect3 = cv2.boundingRect(cnt)
    if best_rect3 is not None:
        x, y, bw, bh = best_rect3
        if (bw * bh) < (h * w) * 0.85:
            return best_rect3

    return None


def crop_and_enhance(image, rect):
    if rect is None:
        h, w = image.shape[:2]
        rect = (0, 0, w, h)
    x, y, w_rect, h_rect = rect
    pad_x = int(w_rect * 0.02)
    pad_y = int(h_rect * 0.02)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w_rect + pad_x)
    y2 = min(image.shape[0], y + h_rect + pad_y)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = image
    enhanced = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))
    kernel_sharpen = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


def is_medicine_package(image, strict=True):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    white_ratio = np.sum(mask_white > 0) / mask_white.size
    if not strict:
        return white_ratio >= 0.02
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_blobs = sum(1 for cnt in contours if 50 < cv2.contourArea(cnt) < 5000)
    mask_blue = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    mask_green = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    nature_ratio = (np.sum(mask_blue > 0) + np.sum(mask_green > 0)) / mask_blue.size
    mask_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask_yellow = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    mask_bright_blue = cv2.inRange(hsv, (100, 100, 100), (120, 255, 255))
    bright_ratio = (np.sum(mask_red > 0) + np.sum(mask_yellow > 0) + np.sum(mask_bright_blue > 0)) / mask_red.size
    has_white = white_ratio >= 0.03
    has_text = text_blobs >= 3
    not_nature = nature_ratio < 0.4
    not_bright = bright_ratio < 0.1
    return has_white and has_text and not_nature and not_bright


# ----------------------------------------------------------------------
# Persiapan Dataset
# ----------------------------------------------------------------------
def prepare_clean_dataset():
    dataset_asli_raw = DATASET_DIR / "asli"
    dataset_palsu_raw = DATASET_DIR / "palsu"
    os.makedirs(DATASET_CLEAN_DIR / "asli", exist_ok=True)
    os.makedirs(DATASET_CLEAN_DIR / "palsu", exist_ok=True)

    total = 0
    failed = 0

    for folder, raw_dir, clean_dir in [
        ('asli', dataset_asli_raw, DATASET_CLEAN_DIR / "asli"),
        ('palsu', dataset_palsu_raw, DATASET_CLEAN_DIR / "palsu")
    ]:
        if not raw_dir.exists():
            print(f"Folder {raw_dir} tidak ditemukan, melewati.")
            continue
        print(f"MEMPROSES FOLDER {folder.upper()}")
        for fname in tqdm(os.listdir(raw_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(raw_dir, fname))
                if img is None:
                    failed += 1
                    continue
                rect = find_medicine_package(img)
                if rect is None:
                    failed += 1
                enhanced = crop_and_enhance(img, rect)
                cv2.imwrite(os.path.join(clean_dir, fname), enhanced)
                total += 1
    print(f"Total diproses: {total}")
    print(f"Gagal deteksi (tapi tetap diproses): {failed}")
    return total, failed, []


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_model_on_clean_dataset():
    dataset_path = DATASET_CLEAN_DIR
    if not dataset_path.exists():
        print(f"Dataset bersih tidak ditemukan di {dataset_path}. Jalankan prepare_clean_dataset() dulu.")
        return None, None

    batch_size = 32
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='binary',
        class_names=['asli', 'palsu']
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='binary',
        class_names=['asli', 'palsu']
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    count_asli = len(os.listdir(DATASET_CLEAN_DIR / "asli"))
    count_palsu = len(os.listdir(DATASET_CLEAN_DIR / "palsu"))
    total = count_asli + count_palsu
    weight_asli = total / (2 * count_asli) if count_asli > 0 else 1.0
    weight_palsu = total / (2 * count_palsu) if count_palsu > 0 else 1.0
    class_weight = {0: weight_asli, 1: weight_palsu}
    print(f"Class weights: {class_weight}")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    print("\n" + "="*60)
    print("TAHAP 1: TRAINING HEAD ONLY")
    print("="*60)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weight,
        callbacks=[early_stop]
    )

    print("\n" + "="*60)
    print("TAHAP 2: FINE-TUNING")
    print("="*60)
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weight,
        callbacks=[early_stop]
    )

    full_history = {}
    for key in history1.history:
        full_history[key] = history1.history[key] + history2.history[key]

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(full_history['accuracy'], label='Train')
    plt.plot(full_history['val_accuracy'], label='Validasi')
    plt.legend(); plt.title('Akurasi')
    plt.subplot(1,2,2)
    plt.plot(full_history['loss'], label='Train')
    plt.plot(full_history['val_loss'], label='Validasi')
    plt.legend(); plt.title('Loss')
    plt.show()

    loss, acc = model.evaluate(val_ds)
    print(f"Validation Accuracy: {acc*100:.2f}%")

    model.save(str(MODEL_DIR / "panadol_v4_clean.keras"))
    print("✅ Model disimpan: panadol_v4_clean.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(MODEL_DIR / "panadol_v4_clean.tflite", 'wb') as f:
        f.write(tflite_model)
    print("✅ Model TFLite disimpan: panadol_v4_clean.tflite")

    return model, full_history


# ----------------------------------------------------------------------
# Inferensi
# ----------------------------------------------------------------------
def predict_medicine(image_path, verify_model, threshold=0.5, confidence_floor=0.80,
                     white_ratio_min=0.03, min_text_blobs=3):
    """Fungsi inferensi asli/palsu dari path gambar."""
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
    """Prediksi + visualisasi."""
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