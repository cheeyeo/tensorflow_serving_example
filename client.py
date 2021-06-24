import tensorflow as tf
import requests
import json
from utils.utils import show
import numpy as np


if __name__ == "__main__":
    _, (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    test_images = test_images.astype("float32") / 255.0

    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    data = json.dumps({"signature": "serving_default", "instances": test_images[0:3].tolist()})

    print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/fashion_models:predict', data=data, headers=headers)

    print(json_response.text)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    predictions = json.loads(json_response.text)['predictions']

    print(predictions)
    for idx, pred in enumerate(predictions):
        title = 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
  class_names[np.argmax(pred)], np.argmax(pred), class_names[test_labels[idx]], test_labels[idx])

        show(test_images, idx, title)