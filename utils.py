import numpy as np
import tensorflow as tf

def expand(x, dim, N,name=None):
    return tf.repeat(tf.expand_dims(x,axis=dim),N,axis=dim,name=name)

def learned_init(units):
    return tf.squeeze(tf.keras.layers.Dense( units,
        activation=None, bias_initializer=None)(tf.ones([1, 1])))

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)