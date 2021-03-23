import keras
import tensorflow as tf

class CustomLambdaLayer(keras.layers.Layer):
    def __init__(self, size=None, is_resize=True):
        super(CustomLambdaLayer, self).__init__()

        self.size = size
        self.is_resize = is_resize

    def call(self, inputs):
        if self.is_resize:
            return self.tf_resize(inputs, self.size)
        else:
            return self.squ_argmax(inputs)

        # return self.function(inputs)

    def tf_resize(self, x, size):
        return tf.compat.v1.image.resize(x, size, method='bilinear', align_corners=True)

    def squ_argmax(self, x):
        x = tf.reshape(x, (1, 512, 512, 21))  # Remove None from batch dim for squeeze

        squeezed_x = tf.squeeze(x)
        print("c_arg_mad squeeze x", squeezed_x.shape)
        arg_max = tf.argmax(squeezed_x, -1)
        print("arg_max shape", arg_max.shape)

        return arg_max