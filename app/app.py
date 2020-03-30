from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import infer
import cv2
import os
import json

import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/ping', methods=['GET'])
def heart_beat():
    return 'PONG'


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)


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
        else:
            return '<h1> bad</h1>'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run()
