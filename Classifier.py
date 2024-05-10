from FeatureExtractor import FeatureExtractor
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np

class Classifier:
    pipeline = None

    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', LinearSVC())
        ])

    def train(self,features,labels):
        self.pipeline.fit(features,labels)
    
    def classify(self,image,method='SIFT'):
        resized_letter = cv2.resize(image, (10, 20), interpolation=cv2.INTER_AREA)
        features = []
        if method == 'SIFT':
            features = FeatureExtractor.applySIFT(resized_letter)
        elif method == 'HOG':
            features = FeatureExtractor.applyHOG(resized_letter)
        predicted_label = self.pipeline.predict(features.reshape(1,-1))[0]
        return predicted_label