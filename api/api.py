from flask import Flask
import requests
import json
import logging
import os


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

# TODO: Predict endpoint
# POST http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]:predict
