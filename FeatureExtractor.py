import cv2
import matplotlib.pyplot as plt
import random
from ImageLoader import ImageLoader
from skimage.feature import hog
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans 
from Segmentation import *
from OrientationDetector import *
from NoiseRemoval import *
from Preprocessing import *


class FeatureExtractor:
    arabic_letters = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'لا', 'م', 'ن', 'ه', 'و', 'ي','لا']
    def __init__(self):
        self.cluster_model = None

    def loadDataset(self, filePath):
        dataSet = []
        labels = []
        for font in range(1,5):
            new_path = filePath + str(font) + "\\"
            images = ImageLoader.loadImages(new_path)
            for i in range(0, len(images)):
                alteredImage = images[i]
                alteredImage = NoiseRemoval.applyMedianFilter(image=images[i],kernel_size=3)
                alteredImage = Segmentation.segment(alteredImage)
                images[i] = alteredImage
            dataSet.extend(images)
            labels.extend((np.full(len(images), str(font))).tolist())
            
        return dataSet,labels
        

    def extractFeatures(self,dataSet,method='HOG'):
        features = []
        for i in range(0,len(dataSet)):
            feature = None
            if method == 'SIFT':
                _a,feature = FeatureExtractor.applySIFT(dataSet[i])
            elif method == 'HOG':
                feature=FeatureExtractor.applyHOG(dataSet[i])
            features.append(feature)
        
        return features
    
    def extract(self,DataSet,method = 'HOG'):
        features = []
        if method == 'SIFT':
            features = self.siftBagOfWords(DataSet)
        elif method == 'HOG':
            features = self.extractFeatures(DataSet)
            
        return features
    
    @staticmethod
    def applyHOG(image):
        fd,hog_image = hog(image,orientations=8,pixels_per_cell=(2,2),cells_per_block=(1,1),visualize=True)
        return fd
    
    @staticmethod
    def applySIFT(image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints,descriptors
    
    # sift feature extraction extension
    def siftBagOfWords(self,dataSet,num_clusters = 360):
        histograms = []
        features = FeatureExtractor.extractFeatures(self,dataSet,method='SIFT')
        filtered_list = [feature for feature in features if feature is not None]
        sift_descriptors = np.concatenate(filtered_list, axis=0)
        if sift_descriptors.shape[1] != 128:
            raise ValueError('Expected SIFT descriptors to have 128 features, got', sift_descriptors.shape[1])
        
        print("Kmeans start")
        self.cluster_model=KMeans(n_clusters=num_clusters)
        self.cluster_model.fit(sift_descriptors)
        for feature in features:
            histogram = np.zeros(num_clusters)
            if (feature is None or len(feature) == 0):
                histograms.append(histogram)
                continue
            
            kmeanLabels = self.cluster_model.predict(feature)
            for label in kmeanLabels:
                histogram[label] += 1
            histograms.append(histogram.astype(float))      
        print("Kmeans Done")            
        return histograms
    
    def siftBagOfWord (self,image,num_clusters = 360):
        keypoints, descriptors = FeatureExtractor.applySIFT(image)
        if descriptors is None:
            return np.zeros(num_clusters)
        else:
            labels = self.cluster_model.predict(descriptors)
            histogram, _ = np.histogram(labels, bins=np.arange(num_clusters + 1))
            return histogram.astype(float)
            
        