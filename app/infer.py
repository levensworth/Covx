import tensorflow.keras as keras
import cv2
import argparse
import numpy as np



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

def get_model(path):
    model = keras.models.load_model(path)
    return model


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
    help="path to input data")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default="covid19.model",
    help="path to the h5 model")
    args = vars(ap.parse_args())

    # load the model
    model = keras.models.load_model(args['model'])
    data = args['data']
    data = cv2.imread(data)
    norm_data = normalize_image(data)
    result = model.predict(norm_data)

    labels = ['negative', 'positive']
    print('result is {}'.format(labels[np.argmax(result)]))

if __name__ == '__main__':
    main()
