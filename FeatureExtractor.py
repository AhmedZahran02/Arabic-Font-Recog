import cv2
import matplotlib.pyplot as plt
import random
from ImageLoader import ImageLoader
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
            selected_numbers = random.sample(range(1, 4000 + 1), 2000)
            letterDataSet = []
            for i in range(0, len(selected_numbers)):
                letter = ImageLoader.loadImage(new_path, str(selected_numbers[i]) + ".png")
                resized_letter = cv2.resize(letter, (10, 20), interpolation=cv2.INTER_AREA)
                letterDataSet.append(resized_letter)
            self.dataSet.append(letterDataSet)

    def extractFeatures(self,method='SIFT'):
        features = []
        labels = []
        for i in range(0,len(self.dataSet)):
            for j in range(0,len(self.dataSet[i])):
                feature = None
                if method == 'SIFT':
                    feature = FeatureExtractor.applySIFT(self.dataSet[i][j])
                elif method == 'HOG':
                    feature=FeatureExtractor.applyHOG(self.dataSet[i][j])
                if feature is not None:
                    features.append(feature)
                    labels.append(str(i+1))
        return features,labels
    
    @staticmethod
    def applyHOG(image):
        fd,hog_image = hog(image,orientations=8,pixels_per_cell=(2,2),cells_per_block=(1,1),visualize=True)
        return fd
    
    @staticmethod
    def applySIFT(image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            return descriptors
        descriptors = np.array(descriptors).flatten()
        padded_discriptors = np.pad(descriptors, (0, 500), 'constant')
        return padded_discriptors[:500]