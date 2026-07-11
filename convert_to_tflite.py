import tensorflow as tf

MODEL_PATH = "models/panadol_v4_clean.keras"
OUTPUT_PATH = "models/panadol_v4_clean.tflite"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("Berhasil!")
print("Output:", OUTPUT_PATH)