from data_generator import DataGenerator
import time
import tensorflow as tf
from keras_resnext_fpn import resnext_fpn

def train():
    print("train")
    training_generator = DataGenerator(base_path="",batch_size=16,  is_val=False)

    epochs = 10

    input_shape = (224, 224, 3)
    nb_labels = 5

    model = resnext_fpn(input_shape=input_shape,
                    nb_labels=nb_labels,
                    depth=(3, 4, 6, 3),
                    cardinality=32,
                    width=4,
                    weight_decay=5e-4,
                    batch_norm=True,
                    batch_momentum=0.9)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        start_time = time.time()
        start_epoch_time = time.time()
        total_reading_time_per_epoch = 0

        for step, (x_batch_train, y_batch_train) in enumerate(training_generator):
            data_gen_batch_reading_time = time.time() - start_time
            total_reading_time_per_epoch += data_gen_batch_reading_time

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # grads = tape.gradient(total_loss, model.trainable_weights)
            # optimizer.apply_gradients(zip(grads, model.trainable_weights))

if __name__ == "__main__":
    train()
