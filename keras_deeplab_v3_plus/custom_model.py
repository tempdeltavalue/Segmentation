import keras
import tensorflow as tf
from model import Deeplabv3

class Deeplabv3Model(keras.Model):
    def __init__(self):
        super(Deeplabv3Model, self).__init__()
        # self.input_shape = (1, 512, 512, 3)
        # self.input_l = Input(batch_shape=(BATCH_SIZE, 224, 224, 3))  # let us say this new InputLayer
        self.base_model = Deeplabv3()

    def call(self, inputs):
        # print("inputs", inputs)
        img_final = tf.cast(inputs, tf.float32) / 127.5

        inputs = img_final - 1.
        # inputs.resize((512, 512, 3))

        ## ConcateLayer
        return self.base_model(inputs)