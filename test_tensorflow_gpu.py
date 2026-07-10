import tensorflow as tf
import time

print(tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0"):
    a = tf.random.normal([10000,10000])
    b = tf.random.normal([10000,10000])

start = time.time()

c = tf.matmul(a,b)

print(time.time()-start)