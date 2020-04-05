import numpy as np
import cv2
import requests
import json
import os
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# set env 

# basic example of using the model
'''
image_path = 'dataset/normal/NORMAL2-IM-0696-0001.jpeg'

data = []
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
data.append(image)

data = np.array(data)/255.0

data = json.dumps({"signature_name": "serving_default", "instances": data.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://18.228.238.49:8501/v1/models/my_model:predict', data=data, headers=headers)
# this is numpy array 0 index represents the probability of infection
# index 1 represents the probability of not being infected
predictions = json.loads(json_response.text)['predictions']
# end example
'''

HOST_ADDRESS = os.getenv('HOST_ADDRESS')
HOST_PORT = os.getenv('HOST_PORT')

def prepare_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image


def load_data_normalized(path):
    '''This functions expects a path where it can find a sufolder for each posible class '''
    sub_folders = os.listdir(path)
    labels  = []
    data = []
    
    for label in sub_folders:
        if os.path.isfile(os.path.join(path, label)):
            continue

        for image_path in os.listdir(os.path.join(path, label)):
            labels.append(label)
            data.append(prepare_image(os.path.join(path, label, image_path)))

    labels = np.array(labels)
    # normalize
    data = np.array(data)/255
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    return data, labels




def predict(data):
    data = json.dumps({"signature_name": "serving_default", "instances": data.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://{}:{}/v1/models/my_model:predict'.format(HOST_ADDRESS, HOST_PORT), data=data, headers=headers)

    return json_response


def main():
    # args = get_args()

    data, labels = load_data_normalized('dataset')
    pred = []
    for img in data:

        json_response = predict(np.array([img]))

        predictions = np.array(json.loads(json_response.text)['predictions'])
        pred.append(predictions[0])

    pred = np.array(pred)
    cm = confusion_matrix(labels.argmax(axis=1), pred.argmax(axis=1))
    print(cm)


if __name__ =='__main__':
    main()