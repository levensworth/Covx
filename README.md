# COVX
### We are searching for a production ready neural net to serve as an alternative method for testing against COVID-19.

*This is a work in progress*: Not ready for production yet.

Based on [this paper](https://pubs.rsna.org/doi/10.1148/radiol.2020200905)
and [this post](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)

Please do not fully trust this model! We are constantly working on improvements.




# How to serve a model ?
## dummy step by step guide.
- Install docker image `docker pull tensorflow/serving`
- `git clone https://github.com/santiagobassani96/Covx.git`
- `cd Covx`
- `MODEL_PATH="$(pwd)/models/servable"`
- `docker run -t --rm -p 8501:8501 -v "$TESTDATA:/models/my_model" -e MODEL_NAME=my_model tensorflow/serving &`
- At this point you should see the model is up and runing
- To use this model:
- Run `./run_server.sh` this will start the flask app to interact with the model.


## Requirements:
* Python3.7
* Tensorflow
* Docker
