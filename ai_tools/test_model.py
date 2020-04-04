import numpy as np
import cv2
import requests
import json

# basic example of using the model
image_path = 'dataset/normal/NORMAL2-IM-0696-0001.jpeg'

data = []
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
data.append(image)

data = np.array(data)/255.0

data = json.dumps({"signature_name": "serving_default", "instances": data.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers)
# this is numpy array 0 index represents the probability of infection
# index 1 represents the probability of not being infected
predictions = json.loads(json_response.text)['predictions']
# end example



