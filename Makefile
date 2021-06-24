.PHONY: start-model

ML_PATH="${PWD}/fashion_models"

start-model:
	docker run -it --rm -p 8501:8501 -v ${ML_PATH}:/models/fashion_models -e MODEL_NAME=fashion_models tensorflow/serving

start-api:
	FLASK_ENV=development \
	FLASK_APP=api/api.py \
	flask run