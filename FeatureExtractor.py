import cv2
import matplotlib.pyplot as plt
import random
from ImageLoader import ImageLoader  # Assuming ImageLoader is implemented in ImageLoader.py
from skimage import data,exposure
from skimage.feature import hog
from sklearn import svm
import numpy as np


class FeatureExtractor:
    arabic_letters = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'لا', 'م', 'ن', 'ه', 'و', 'ي']

    def __init__(self):
        self.dataSet = []

    def loadDataset(self, filePath):
        for character in range(1,30):
            new_path = filePath + str(character) + "\\"
            selected_numbers = random.sample(range(1, 4000 + 1), 500)
            letterDataSet = []
            for i in range(0, len(selected_numbers)):
                letter = ImageLoader.loadImage(new_path, str(selected_numbers[i]) + ".png")
                resized_letter = cv2.resize(letter, (20, 40), interpolation=cv2.INTER_AREA)
                letterDataSet.append(resized_letter)
            self.dataSet.append(letterDataSet)

    def extractFeatures(self):
        features = []
        for i in range(0,len(self.dataSet)):
            for j in range(0,len(self.dataSet[i])):
                features.append(FeatureExtractor.applySIFT(self.dataSet[i][j]))
        return features
    
    @staticmethod
    def applyHOG(image):
        fd,hog_image = hog(image,orientations=8,pixels_per_cell=(2,2),cells_per_block=(1,1),visualize=True)
        #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return fd
    
    @staticmethod
    def applySIFT(image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors = [descriptor for descript in descriptors for descriptor in descript]
            return descriptors
        else:
            return []