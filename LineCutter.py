import numpy as np
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class LineCutter:
    @staticmethod
    def extractLines(image):
        horizontalHist = np.sum(image, axis=1)
        hPeaks, _ = find_peaks(horizontalHist,height= np.max(horizontalHist) * 0.5, distance=10)
        
        halfLine = 0
        i = hPeaks[0] - 1
        while i >= 0 and horizontalHist[i] != 0:
            halfLine += 1
            i -= 1
            
        lines = []
    
        for peak in hPeaks:
            line = image[peak-halfLine:peak+halfLine,0:image.shape[1]]
            lines.append(line)

        return lines