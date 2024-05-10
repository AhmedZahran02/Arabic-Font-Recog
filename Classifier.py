from FeatureExtractor import FeatureExtractor
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from ImageLoader import * 

class Classifier:
    pipeline = None

    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('gradient_boosting', GradientBoostingClassifier())
        ])

    def train(self,features,labels):
        self.pipeline.fit(features,labels)
    
    def classify(self,image,featureExtractor,method='SIFT'):
        resized_letter = cv2.resize(image, (10, 20), interpolation=cv2.INTER_AREA)
        # _, resized_letter = cv2.threshold(resized_letter, 0, 255, cv2.THRESH_BINARY)
        # ImageLoader.print(resized_letter)
        features = []
        if method == 'SIFT':
            features = featureExtractor.siftBagOfWord(image=resized_letter)
        elif method == 'SURF':
            features = featureExtractor.surfBagOfWord(image=resized_letter)
        elif method == 'HOG':
            features = FeatureExtractor.applyHOG(resized_letter)
        elif method == 'COMBINED':
            siftFeatures = featureExtractor.siftBagOfWord(image=resized_letter)
            hogFeatures = FeatureExtractor.applyHOG(resized_letter)
            features = np.concatenate((siftFeatures, hogFeatures))
        
        predicted_label = self.pipeline.predict(features.reshape(1,-1))[0]
        return predicted_label