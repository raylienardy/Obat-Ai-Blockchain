import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("models/panadol_v4_clean.keras")
print("✓ Model loaded")

print("Creating converter...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

print("Converting...")
tflite_model = converter.convert()

print("Saving...")
with open("test.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Berhasil")