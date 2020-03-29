from flask import Flask, request
import infer
import cv2
import os
import json

import numpy as np

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def heart_beat():
    return 'PONG'

@app.route('/predict', methods=['POST'])
def predict():
    model = infer.get_model('models/covid19.model')
    # read image directly from post request
    # https://stackoverflow.com/questions/58082051/how-to-convert-image-file-object-to-numpy-array-in-with-opencv-python
    file_data = request.files['file'].read()
    data = cv2.imdecode(np.fromstring(file_data, np.uint8), cv2.IMREAD_COLOR)

    norm_data = infer.normalize_image(data)
    # result is [postive_prob , negative_prob]
    result = model.predict(norm_data)[0]
    # create result
    result = {
        "positive": result[0],
        "negative": result[1]
    }
    return  result.__str__()

if __name__ == '__main__':
    app.run()
