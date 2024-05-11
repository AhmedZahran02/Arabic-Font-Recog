import cv2
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import os
import numpy as np
from xgboost import XGBClassifier
from cv2.xfeatures2d import SIFT_create as sift_create
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import sys
import time

import main as utils


data_path = sys.argv[1]
print("Loading Images Paths.. ")
images_paths = []
for img_path in os.listdir(data_path):
    images_paths.append(int(img_path[:-5])) # remove the .jpeg
images_paths.sort()

print(f"Loaded: {len(images_paths)} images.")

print("Beginning test.")
timimg_file = open("time.txt" , 'w')
results_file = open("results.txt" , 'w')

for img_path in images_paths:
    # load the image
    img = cv2.imread(os.path.join(data_path , f"{str(img_path)}.jpeg"))
    start = time.time()
    prediction = utils.ClassifyImage(img)
    delta = time.time() - start
    print(f"Prediction for: \"{img_path}.jpeg\" = {prediction}  , in {delta:.10f} seconds")
    timimg_file.write(f"{delta:.3f}\n")
    results_file.write(f"{4 - int(prediction)}\n")

timimg_file.close()
results_file.close()
print("Test ended")