from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
import tensorflow as tf
import requests
import json
import logging
import os
import numpy as np


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

logger = logging.getLogger("api.api")

tfx_host = os.environ.get("TFX_SERVER", "localhost")
tfx_url = "http://{}:8501/v1/models".format(tfx_host)

app = Flask(__name__)

# Model Status API
# GET http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]

@app.route("/models/<model_name>", methods=["GET"])
@app.route("/models/<model_name>/versions/<version>", methods=["GET"])
@app.route("/models/<model_name>/labels/<label>", methods=["GET"])
def get_model_status(model_name, version=None, label=None):
    app.logger.info("GOT MODEL NAME: {}".format(model_name))
    url = "{}/{}".format(tfx_url, model_name)
    if version:
        url = "{}/versions/{}".format(url, version)
    if label:
        url = "{}/labels/{}".format(url, label)

    headers = {"content-type": "application/json"}
    resp = requests.get(url, headers=headers)
    resp = json.loads(resp.text)
    return resp

# Model Metadata API
# GET http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]/metadata

@app.route("/models/<model_name>/metadata", methods=["GET"])
@app.route("/models/<model_name>/versions/<version>/metadata", methods=["GET"])
@app.route("/models/<model_name>/labels/<label>/metadata", methods=["GET"])
def get_model_metadata(model_name, version=None, label=None):
    app.logger.info("GOT MODEL NAME: {}".format(model_name))
    url = "{}/{}".format(tfx_url, model_name)
    if version:
        url = "{}/versions/{}".format(url, version)
    if label:
        url = "{}/labels/{}".format(url, label)
    url = "{}/metadata".format(url)

    headers = {"content-type": "application/json"}
    resp = requests.get(url, headers=headers)
    resp = json.loads(resp.text)
    return resp


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, (28, 28))

    label = tf.strings.split(img_path, "_")[-1]
    label = tf.strings.split(label, ".")[-2]
    label = tf.strings.to_number(label, tf.int32)
    return img, label


# TODO: Predict endpoint
# POST http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]:predict
@app.route("/models/<model_name>/predict", methods=["POST"])
def get_prediction(model_name):
    inference_dir = os.path.join(os.getcwd(), "infer")
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    if request.method == "POST":
        f = request.files["data"]
        img_path = os.path.join(inference_dir, secure_filename(f.filename))
        f.save(img_path)
        
        img, label = load_image(img_path)
        img = img.numpy()
        label = label.numpy()
        print(np.min(img))
        print(np.max(img))

        if not(np.min(img) == 0.0 and np.max(img) == 1.0):
            img = img / 255.0

        # turn into batch of 1
        img = np.expand_dims(img, axis=0)

        # make actual prediction
        data = json.dumps({"signature": "serving_default", "instances": img.tolist()})

        headers = {"content-type": "application/json"}

        json_response = requests.post('http://localhost:8501/v1/models/fashion_models:predict', data=data, headers=headers)
        print(json_response.text)
        preds = json.loads(json_response.text)["predictions"]
        resp = "The model thought this was a {} (class {}), and it was actually a {} (class {})".format(class_names[np.argmax(preds)], np.argmax(preds), class_names[label], label)

        return resp