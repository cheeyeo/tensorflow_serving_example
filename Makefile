.PHONY: start-model start-api saved-model

ML_PATH="${PWD}/fashion_models"

saved-model:
	saved_model_cli show --dir ./fashion_models/1 --all

start-model:
	docker run -it --rm -p 8501:8501 -p 8500:8500 -v ${ML_PATH}:/models/fashion_models -e MODEL_NAME=fashion_models tensorflow/serving

start-api:
	FLASK_ENV=development \
	FLASK_APP=api/api.py \
	flask run