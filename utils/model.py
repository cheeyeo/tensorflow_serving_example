# Utils for model training / inference
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis.predict_pb2 import PredictRequest
import grpc
import requests
import json

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def grpc_request(tfx_host, model_name, img):
    """
    Makes a grpc request to the grpc endpoint

    Returns a numpy response
    """
    input_name = "conv2d_1_input"
    output_name = "fc"

    request = PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = "serving_default"
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(img))

    # send request to server
    channel = grpc.insecure_channel(tfx_host)
    predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    response = predict_service.Predict(request, timeout=10.0)
    preds = response.outputs[output_name]
    preds = tf.make_ndarray(preds)

    return preds


def model_metadata(url, model_name, version=None, label=None):
    url = "{}/{}".format(url, model_name)
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