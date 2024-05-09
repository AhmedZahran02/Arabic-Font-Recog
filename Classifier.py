from FeatureExtractor import FeatureExtractor
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
import numpy as np
class Classifier:
    clf = None

    def preprocess(num_of_labels,number_of_feature_vectors):
        label_flattened = []
        for i in range(0,num_of_labels):
            label_flattened = label_flattened + (np.full((number_of_feature_vectors,),i)).tolist()
        return label_flattened

    def __init__(self):
        self.clf = SVC(kernel='linear')

    def train(self,features):
        #for i in range(0,len(features)):
            #for j in range(0,len(features[i])):
                # print(len(features[i][j]))
                # print(len(np.full((len(features[i][j]),),i)))
        # self.clf.fit([element for sublist in features for element in sublist],Classifier.preprocess(len(features),len(features[0])))
        print(len(features))
        self.clf.fit(features,Classifier.preprocess(29,500))

    
    def classify(self,image):
        resized_letter = cv2.resize(image, (20, 40), interpolation=cv2.INTER_AREA)
        features = FeatureExtractor.applySIFT(resized_letter)
        predicted_label = self.clf.predict(features.reshape(-1,1))[0]
        return predicted_label