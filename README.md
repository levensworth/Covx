# COVX
### we are searching for a production ready neural net to serve as an alternative method for testing against COVID-19.

*This is a work in progress*: Not ready for production yet.

Based on [this paper] (https://pubs.rsna.org/doi/10.1148/radiol.2020200905)
ans [this post] (https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)

Please do not fully trust this model! we are constntly working on improvments.




# How to serve a model ?
## dummy setp by setp guide.
- Install docker image `docker pull tensorflow/serving`
- `git clone https://github.com/santiagobassani96/Covx.git`
- `cd Covx`
- `MODEL_PATH="$(pwd)/models/servable"`
- `docker run -t --rm -p 8501:8501 -v "$TESTDATA:/models/my_model" -e MODEL_NAME=my_model tensorflow/serving &`
- At this poit you should see the model is up and runing


## Requirements:
* Python3.7
* Tensorflow
* Docker
