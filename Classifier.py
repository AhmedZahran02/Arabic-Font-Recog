from FeatureExtractor import FeatureExtractor
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from ImageLoader import * 

class Classifier:
    pipeline = None

    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf'))
        ])

    def train(self,features,labels):
        self.pipeline.fit(features,labels)
    
    def classify(self,image,featureExtractor,method='SIFT'):
        # resized_letter = cv2.resize(image, (10, 20), interpolation=cv2.INTER_AREA)
        # _, resized_letter = cv2.threshold(resized_letter, 0, 255, cv2.THRESH_BINARY)
        ImageLoader.print(image)
        features = []
        if method == 'SIFT':
            features = featureExtractor.bagOfWord(image=image)
        elif method == 'HOG':
            features = FeatureExtractor.applyHOG(image)
        elif method == 'COMBINED':
            siftFeatures = featureExtractor.bagOfWord(image=image)
            hogFeatures = FeatureExtractor.applyHOG(image)
            features = np.concatenate((siftFeatures, hogFeatures))
        
        predicted_label = self.pipeline.predict(features.reshape(1,-1))[0]
        return predicted_label