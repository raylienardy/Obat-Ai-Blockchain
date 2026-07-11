import tensorflow as tf
from utils.trainer import train_model_on_clean_dataset

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

train_model_on_clean_dataset()