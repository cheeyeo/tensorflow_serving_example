# Get some sample images to test predictions
import tensorflow as tf
import numpy as np
import argparse
import os


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=5, help="Num of sample images to generate")
    args = vars(ap.parse_args())

    _, (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

    print(testX.shape)
    print(testY.shape)

    if not os.path.exists("testimgs"):
        os.mkdir("testimgs")

    sampled_images = testX[:args["num"]]
    sampled_labels = testY[:args["num"]]
    for idx, img in enumerate(sampled_images):
        print(sampled_labels[idx])
        print(img.shape)
        print(img.dtype)
        img = np.expand_dims(img, axis=-1)
        fpath = os.path.join(os.getcwd(), "testimgs", "test_{}_label_{}.png".format(idx, sampled_labels[idx]))
        tf.keras.preprocessing.image.save_img(
            fpath, img, data_format="channels_last", scale=False
        )