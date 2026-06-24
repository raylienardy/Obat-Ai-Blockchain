import argparse
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi
IMG_HEIGHT, IMG_WIDTH = 224, 224
THRESHOLD = 0.90

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, '..', 'database', 'feature_database.pkl')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'feature_extractor.h5')

# Load
print("Memuat model...")
feature_extractor = load_model(MODEL_PATH)
print("Memuat database...")
with open(DATABASE_PATH, 'rb') as f:
    data = pickle.load(f)
database_features = data['features']
database_names = data['names']
print(f"Database: {database_features.shape[0]} referensi")

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

def verify_single(img_path, threshold=THRESHOLD):
    query_feat = extract_features(img_path, feature_extractor)
    similarities = cosine_similarity([query_feat], database_features)[0]
    max_sim = np.max(similarities)
    top5_idx = np.argsort(similarities)[-5:][::-1]
    return {
        'max_similarity': max_sim,
        'top5_avg': np.mean(similarities[top5_idx]),
        'top_matches': [(database_names[i], similarities[i]) for i in top5_idx],
        'is_asli': max_sim >= threshold,
        'threshold': threshold
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tet-Suv Verifikasi Keaslian Obat')
    parser.add_argument('input', help='Path ke gambar atau folder')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Threshold (default 0.90)')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        res = verify_single(args.input, args.threshold)
        print(f"Gambar: {args.input}")
        print(f"  Max Similarity: {res['max_similarity']*100:.2f}%")
        print(f"  Rata-rata Top 5: {res['top5_avg']*100:.2f}%")
        print(f"  Threshold: {args.threshold*100:.0f}%")
        print(f"  Hasil: {'✅ ASLI' if res['is_asli'] else '❌ PALSU'}")
        print("  5 Termirip:")
        for name, sim in res['top_matches']:
            print(f"    - {name}: {sim*100:.2f}%")
    elif os.path.isdir(args.input):
        files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if not files:
            print("Tidak ada gambar di folder ini.")
        else:
            for fname in sorted(files):
                path = os.path.join(args.input, fname)
                res = verify_single(path, args.threshold)
                status = '✅ ASLI' if res['is_asli'] else '❌ PALSU'
                print(f"{fname:<30} Max: {res['max_similarity']*100:.2f}% -> {status}")
    else:
        print("Input tidak valid.")