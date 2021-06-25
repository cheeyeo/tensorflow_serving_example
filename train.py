import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import subprocess


if __name__ == "__main__":
    print("Loading dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("Train images shape: {}, type: {}".format(train_images.shape, train_images.dtype))
    print("Test images shape: {}, type: {}".format(test_images.shape, test_images.dtype))

    model = Sequential([
        Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", input_shape=(28, 28, 1), name="conv2d_1"),
        Flatten(),
        Dense(10, name="fc")
    ])

    model.summary()

    print("Evaluating model...")
    epochs = 5
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test acc: {:.2f}, Test loss: {:.2f}".format(test_acc, test_loss))

    # save model in SavedModel format
    version = 1
    model_path = os.path.join("fashion_models", str(version))
    tf.keras.models.save_model(
        model,
        model_path,
        overwrite=True,
        include_optimizer=True,
    )