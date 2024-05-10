import cv2
import matplotlib.pyplot as plt
import random
from ImageLoader import ImageLoader
from skimage.feature import hog
import numpy as np
from sklearn.cluster import KMeans



class FeatureExtractor:
    arabic_letters = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'لا', 'م', 'ن', 'ه', 'و', 'ي']
    def __init__(self):
        self.dataSet = []
        self.kmeans = None

    def loadDataset(self, filePath):
        for character in range(1,30):
            new_path = filePath + str(character) + "\\"
            selected_numbers = random.sample(range(1, 4000 + 1), 500)
            letterDataSet = []
            for i in range(0, len(selected_numbers)):
                letter = ImageLoader.loadImage(new_path, str(selected_numbers[i]) + ".png")
                resized_letter = cv2.resize(letter, (10, 20), interpolation=cv2.INTER_AREA)
                # _, resized_letter = cv2.threshold(resized_letter, 0, 255, cv2.THRESH_BINARY)
                letterDataSet.append(resized_letter)
            self.dataSet.append(letterDataSet)

    def extractFeatures(self,method='HOG'):
        features = []
        labels = []
        for i in range(0,len(self.dataSet)):
            for j in range(0,len(self.dataSet[i])):
                feature = None
                if method == 'SIFT':
                    _a,feature = FeatureExtractor.applySIFT(self.dataSet[i][j])
                elif method == 'HOG':
                    feature=FeatureExtractor.applyHOG(self.dataSet[i][j])
                elif method == 'SURF':
                    _a,feature = FeatureExtractor.applySURF(self.dataSet[i][j])
                if feature is not None:
                    features.append(feature)
                    labels.append(str(i+1))
        
        return features,labels
    
    def extract(self,method = 'HOG'):
        features = []
        labels = []
        if method == 'SIFT':
            features,labels = self.siftBagOfWords()
        elif method == 'HOG':
            features,labels = self.extractFeatures()
        elif method == 'SURF':
            features,labels = self.surfBagOfWords()
            
        return features,labels
    
    @staticmethod
    def applyHOG(image):
        fd,hog_image = hog(image,orientations=8,pixels_per_cell=(2,2),cells_per_block=(1,1),visualize=True)
        return fd
    
    @staticmethod
    def applySIFT(image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints,descriptors
    
    @staticmethod
    def applySURF(image):
        surf = cv2.xfeatures2d.SURF_create(400)
        keypoints, descriptors = surf.detectAndCompute(image, None)
        return keypoints, descriptors
    
    # sift feature extraction extension
    def siftBagOfWords(self,num_clusters = 20):
        histograms = []
        features,labels = FeatureExtractor.extractFeatures(self,method='SIFT')
        sift_descriptors = np.concatenate(features, axis=0)
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.kmeans.fit(sift_descriptors)
        
        for i in range(0,len(self.dataSet)):
            for j in range(0,len(self.dataSet[i])):
                keypoints, descriptors = FeatureExtractor.applySIFT(self.dataSet[i][j])
                if descriptors is not None:
                    kmeanLabels = self.kmeans.predict(descriptors)
                    histogram, _ = np.histogram(kmeanLabels, bins=np.arange(num_clusters + 1))
                    histograms.append(histogram.astype(float))
                    
        return histograms,labels
    
        # sift feature extraction extension
    def surfBagOfWords(self,num_clusters = 20):
        histograms = []
        features,labels = FeatureExtractor.extractFeatures(self,method='SURF')
        surf_descriptors = np.concatenate(features, axis=0)
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.kmeans.fit(surf_descriptors)
        
        for i in range(0,len(self.dataSet)):
            for j in range(0,len(self.dataSet[i])):
                keypoints, descriptors = FeatureExtractor.applySURF(self.dataSet[i][j])
                if descriptors is not None:
                    kmeanLabels = self.kmeans.predict(descriptors)
                    histogram, _ = np.histogram(kmeanLabels, bins=np.arange(num_clusters + 1))
                    histograms.append(histogram.astype(float))
                    
        return histograms,labels
    
    def combinedFeatureExtraction(self,num_clusters = 20):
        combinedFeatures = []
        features,labels = FeatureExtractor.extractFeatures(self,method='SIFT')
        sift_descriptors = np.concatenate(features, axis=0)
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.kmeans.fit(sift_descriptors)
        
        for i in range(0,len(self.dataSet)):
            for j in range(0,len(self.dataSet[i])):
                keypoints, descriptors = FeatureExtractor.applySIFT(self.dataSet[i][j])
                hogFeature = FeatureExtractor.applyHOG(self.dataSet[i][j])
                if descriptors is not None:
                    kmeanLabels = self.kmeans.predict(descriptors)
                    histogram, _ = np.histogram(kmeanLabels, bins=np.arange(num_clusters + 1))
                    combinedFeatures.append(np.concatenate((histogram.astype(float), hogFeature)))
                    
        return combinedFeatures,labels
    
    def siftBagOfWord (self,image,num_clusters = 20):
        keypoints, descriptors = FeatureExtractor.applySIFT(image)
        if descriptors is None:
            return np.zeros(num_clusters)
        else:
            labels = self.kmeans.predict(descriptors)
            histogram, _ = np.histogram(labels, bins=np.arange(num_clusters + 1))
            return histogram.astype(float)
        
    def surfBagOfWord (self,image,num_clusters = 20):
        keypoints, descriptors = FeatureExtractor.applySURF(image)
        if descriptors is None:
            return np.zeros(num_clusters)
        else:
            labels = self.kmeans.predict(descriptors)
            histogram, _ = np.histogram(labels, bins=np.arange(num_clusters + 1))
            return histogram.astype(float)
            
        