import argparse

def get_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default="covid19.model",
        help="path to output loss/accuracy plot")
    return vars(ap.parse_args())

