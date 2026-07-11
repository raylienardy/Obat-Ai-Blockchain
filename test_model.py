import tensorflow as tf

model = tf.keras.models.load_model("models/panadol_v4_clean.keras")

print(model.summary())