# Pip install the client:
# pip install clarifai

from clarifai.rest import ClarifaiApp
import pprint
# Create your API key in your account's Application details page:
# https://clarifai.com/apps

app = ClarifaiApp(api_key='c5f139adf6934ddcb7f638d391256318')
models = app.models
model = app.models.covid_1
response = model.predict_by_filename('dataset/covid/1-s2.0-S0140673620303706-fx1_lrg.jpg')
# You could also use model.predict_by_bytes or model.predict_by_base64
pprint.pprint(response)
