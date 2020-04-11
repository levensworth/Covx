import numpy as np 

from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize
import matplotlib.pyplot as plt
import os


'''
The idea of this script is to test a bluriness clasifier for PNG photos, given aor goal to move from png to a more precise extesion as DICOM.
This is will be explore as a possible side dev.

currentl what we doo is plot the max and var of each image to see if there is a clear distinction between blurn and clear photos. Which could indicate 
the possibility to use a classifier.
'''

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	help="path to output loss/accuracy plot")
# ap.add_argument("-m", "--model", type=str, default="covid19.model",
# 	help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())



# Load image
path = f'dataset'
args = {}
args['dataset'] = path
maximums = []
variances = []
labels = os.listdir(args['dataset'])
for label in labels:
    paths = os.listdir(os.path.join(args['dataset'], label, label))
    for image_path in paths:
        img = io.imread(path)
        
        # Resize image
        img = resize(img, (224, 224))

        # Grayscale image
        img = rgb2gray(img)

        # Edge detection
        edge_laplace = laplace(img, ksize=3)
        
        # variance
        variance(edge_laplace)

        # Maximum
        np.amax(edge_laplace)


plt.plot(variances, maximums)

# Print output
print(f"Variance: {variance(edge_laplace)}")
print(f"Maximum : {np.amax(edge_laplace)}")