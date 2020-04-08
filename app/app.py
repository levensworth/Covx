from flask import Flask, flash, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import cv2
import os
import json
import numpy as np
import requests


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/ping', methods=['GET'])
def heart_beat():
    return 'PONG'


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
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
            print('[INFO] reading file ...')
            # read image directly from post request
            # https://stackoverflow.com/questions/58082051/how-to-convert-image-file-object-to-numpy-array-in-with-opencv-python
            file_data = request.files['file'].read()
            data = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            norm_data = normalize_image(data)
            # result is [postive_prob , negative_prob]
            try:

                json_response = serve_prediciton(norm_data)
                predictions = np.array(json.loads(json_response.text)['predictions'])
                # create result
                result = {
                    "positive": predictions[0,0],
                    "negative": predictions[0,1]
                }

                response = app.response_class(
                    response=json.dumps(result),
                    status=200,
                    mimetype='application/json'
                )
                return response


            except requests.exceptions.HTTPError as errh:
                print ("Http Error:",errh)
                return Response(errh, status=429)

            except requests.exceptions.ConnectionError as errc:
                print ("Error Connecting:",errc)
                return Response(errc, status=503)

            except requests.exceptions.Timeout as errt:
                print ("Timeout Error:",errt)
                return Response(errt, status=408)

            except requests.exceptions.RequestException as err:
                print ("OOps: Something Else",err)
                return Response(err, status=507)

            except Exception as e:
                print('[ERROR] {}'.format(e))
                return Response(e, status=400)
        else:
            return Response('Not supported', status=415)
    return Response(None, status=403)

def normalize_image(data):
    # normalize input
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = data
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    inputX = []
    inputX.append(image)
    inputX = np.array(inputX)/255.0
    return inputX

def serve_prediciton(data):
    print('[INFO] going to ask the Model ...')
    HOST_ADDRESS = os.getenv('HOST_ADDRESS')
    HOST_PORT = os.getenv('HOST_PORT')
    try:
        data = json.dumps({"signature_name": "serving_default", "instances": data.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post('http://{}:{}/v1/models/my_model:predict'.format(HOST_ADDRESS, HOST_PORT), data=data, headers=headers)

        return json_response
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
        raise errh
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
        raise errc
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
        raise errt
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)
        raise err

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
