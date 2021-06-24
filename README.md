### FLASK, GRPC, Docker Tensorflow serving

Example of training and serving a simple sequential model in TF using TFX in docker

Also create a simple FLASK API endpoint to query model's metadata and for predictions


### Running

Build and train model first using `train.py`. This builds and trains a very basic convnet using fashion mnist dataset

Run `make start-model` to start TFX server

Run `make start-api` to start api endpoint

Query the api for model's status:
```
curl -XGET http://localhost:5000/model/fashion_models
```

Query the api for model's metadata:
```
curl -XGET http://localhost:5000/model/fashion_models/metadata
```

To test a prediction, use `client.py`

The predict endpoint is still WIP.


### Running TFX in docker

Different port numbers expose different prediction types:
```
GRPC => 8500
Rest => 8501 
```

To run locally using docker:
```
 docker run -it --rm -p 8501:8501 -v "$ML_PATH:/models/fashion_models" -e MODEL_NAME=fashion_models tensorflow/serving
```

### References
[Simple REST example](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)

[TFX Docker](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)

[Client API](https://www.tensorflow.org/tfx/serving/api_rest)