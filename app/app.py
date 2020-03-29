from flask import Flask, request
import infer
import cv2
import os
import json

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def heart_beat():
    return 'PONG'

@app.route('/predict', methods=['POST'])
def predict():
    model = infer.get_model('models/covid19.model')
    file_data = request.files['file']
    file_data.save('app/file')

    data = cv2.imread('app/file')
    norm_data = infer.normalize_image(data)
    result = model.predict(norm_data)[0]
    result = {
        "positive": result[0],
        "negative": result[1]
    }
    return  result.__str__()
if __name__ == '__main__':
    app.run()
