"""
Training model classifier MobileNetV2.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from .config import IMG_HEIGHT, IMG_WIDTH, DATASET_CLEAN_DIR, MODEL_DIR, MODEL_FILENAME, TFLITE_FILENAME


def train_model_on_clean_dataset():
    dataset_path = DATASET_CLEAN_DIR
    if not dataset_path.exists():
        print(f"Dataset bersih tidak ditemukan di {dataset_path}. Jalankan prepare_clean_dataset() dulu.")
        return None, None

    print("STEP 1 - Loading dataset")
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
    
    print("STEP 2 - Dataset loaded")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    
    print("STEP 3 - Augmentation created")

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

    print("STEP 4 - Building MobileNetV2")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    print("STEP 5 - Model built")
    
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    print("STEP 5 - Model built")

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
    
    
    print("STEP 6 - Model compiled")

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

    model.save(str(MODEL_DIR / MODEL_FILENAME))
    print(f"✅ Model disimpan: {MODEL_FILENAME}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(MODEL_DIR / TFLITE_FILENAME, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Model TFLite disimpan: {TFLITE_FILENAME}")

    return model, full_history