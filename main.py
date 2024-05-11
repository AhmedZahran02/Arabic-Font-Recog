import cv2
import numpy as np
import pickle
import bz2file as bz2
from scipy.signal import find_peaks

clf = None
Kmean = None

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def applyMedianFilter(image, kernel_size=5):
    filteredImage = cv2.medianBlur(image, kernel_size)
    return filteredImage

def segment(image):
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0,256))
        peaks, _ = find_peaks(hist, height=0, distance=10)
        topTwoPeaks = sorted(sorted(peaks, key=lambda x: -hist[x])[:2],reverse=True)
        
        if len(topTwoPeaks) == 0:
            topTwoPeaks.append(255)
            topTwoPeaks.append(0)
        if len(topTwoPeaks) < 2:
            topTwoPeaks.append(topTwoPeaks[0])
            
        threshold = (topTwoPeaks[0]+topTwoPeaks[1])/2.0
        if abs(topTwoPeaks[0] - image[0][0]) < abs(topTwoPeaks[1] - image[0][0]):
            threshold += abs(topTwoPeaks[0]-topTwoPeaks[1])/4
            _, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        else :
            threshold -= abs(topTwoPeaks[0]-topTwoPeaks[1])/4
            _, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)       
        
        return segmented_image

def applySIFT(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints,descriptors
  
def siftBagOfWord (image,num_clusters = 360):
    keypoints, descriptors = applySIFT(image)
    if descriptors is None:
        return np.zeros(num_clusters)
    else:
        labels = Kmean.predict(descriptors)
        histogram, _ = np.histogram(labels, bins=np.arange(num_clusters + 1))
        return histogram.astype(float)
    
def classify(image):

    features = siftBagOfWord(image=image)
    predicted_label = clf.predict(features.reshape(1,-1))[0]
    return predicted_label


clf = decompress_pickle('models\\classifier2.pbz2')
if clf is not None:
    print("Classifier Loaded")
else:
    print("Failed to Load Classifier") 
    
with open('models/kmean.pkl', 'rb') as f:
    Kmean = pickle.load(f)
if Kmean is not None:
    print("Kmean Loaded")
else:
    print("Failed to Load Kmean") 


def ClassifyImage(image):
    alteredImage = applyMedianFilter(image,kernel_size=3)
    alteredImage = segment(alteredImage)
    result = classify(alteredImage)
    return result



