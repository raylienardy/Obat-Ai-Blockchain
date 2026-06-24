import os
import numpy as np
import pickle
import tensorflow as tf  # <-- TAMBAHKAN BARIS INI
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# ================= KONFIGURASI =================
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Path relatif dari folder scripts/
DATASET_ASLI_DIR = os.path.join('..', 'dataset', 'asli')
MODEL_SAVE_PATH = os.path.join('..', 'models', 'feature_extractor.h5')
DATABASE_OUTPUT = os.path.join('..', 'database', 'feature_database.pkl')
# ===============================================

# 1. Load MobileNetV2 sebagai feature extractor (tanpa top layer)
print("Memuat MobileNetV2 (pre-trained ImageNet)...")
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
feature_extractor.save(MODEL_SAVE_PATH)
print(f"Model disimpan di: {MODEL_SAVE_PATH}")

# 2. Fungsi untuk ekstrak fitur
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

# 3. Proses semua gambar di folder asli
print(f"Mencari gambar di: {DATASET_ASLI_DIR}")
asli_files = [f for f in os.listdir(DATASET_ASLI_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(asli_files) == 0:
    raise FileNotFoundError(f"Tidak ada gambar ditemukan di {DATASET_ASLI_DIR}. Pastikan folder tersebut berisi file .jpg/.jpeg/.png")

print(f"Ditemukan {len(asli_files)} gambar. Mulai ekstraksi fitur...")

features_list = []
names = []

for fname in tqdm(asli_files):
    path = os.path.join(DATASET_ASLI_DIR, fname)
    feat = extract_features(path, feature_extractor)
    features_list.append(feat)
    names.append(fname)

# 4. Simpan database fitur
os.makedirs(os.path.dirname(DATABASE_OUTPUT), exist_ok=True)
database = {
    'features': np.array(features_list),
    'names': names
}

with open(DATABASE_OUTPUT, 'wb') as f:
    pickle.dump(database, f)

print(f"Database berisi {len(features_list)} fitur berhasil disimpan di: {DATABASE_OUTPUT}")
print("Selesai!")