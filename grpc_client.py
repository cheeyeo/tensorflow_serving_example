from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis.predict_pb2 import PredictRequest
import tensorflow as tf
import numpy as np
import grpc
import argparse
import os
from utils.model import load_image
from utils.model import class_names
from utils.model import model_metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="Path to input image")
    args = vars(ap.parse_args())

    print("Prediction using grpc service...")
    input_name = "conv2d_1_input"
    output_name = "fc"

    img, label = load_image(args["image"])
    img = img.numpy()
    img = np.expand_dims(img, axis=0)
    label = label.numpy()
    print("Input image shape: ", img.shape)
    print("Expected Label: ", label)

    request = PredictRequest()
    request.model_spec.name = "fashion_models"
    request.model_spec.signature_name = "serving_default"
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(img))

    # send request to server
    channel = grpc.insecure_channel("localhost:8500")
    predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    response = predict_service.Predict(request, timeout=10.0)

    preds = response.outputs[output_name]
    preds = tf.make_ndarray(preds)
    print(type(preds))
    print(preds)

    res = np.argmax(preds)
    expected_label = class_names[label]
    predicted_label = class_names[res]
    print("Expected Label > ", expected_label)
    print("Predicted Label > ", predicted_label)