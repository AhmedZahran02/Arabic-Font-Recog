import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class Segmentation:
    @staticmethod
    def segment(image):
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0,256))
        peaks, _ = find_peaks(hist, height=0, distance=10)
        topTwoPeaks = sorted(sorted(peaks, key=lambda x: -hist[x])[:2],reverse=True)
        threshold = (topTwoPeaks[0]+topTwoPeaks[1])/2.0
        if abs(topTwoPeaks[0] - image[0][0]) < abs(topTwoPeaks[1] - image[0][0]):
            threshold += abs(topTwoPeaks[0]-topTwoPeaks[1])/4
            _, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        else :
            threshold -= abs(topTwoPeaks[0]-topTwoPeaks[1])/4
            _, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)       
        
        return segmented_image