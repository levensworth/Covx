MODEL_PATH="$(pwd)/models/servable"
docker run -t --rm -p 8501:8501 -v "$MODEL_PATH:/models/my_model" -e MODEL_NAME=my_model tensorflow/serving &