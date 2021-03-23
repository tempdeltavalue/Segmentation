import keras
import tensorflow as tf

class CustomNormalizationLayer(keras.layers.Layer):
    def __init__(self, size=None, is_resize=True):
        super(CustomNormalizationLayer, self).__init__()

    def call(self, inputs):
        mean_subtraction_value = 127.5

        return inputs #(inputs / mean_subtraction_value) - 1.
