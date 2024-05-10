from scipy.signal import find_peaks
import numpy as np
import cv2
import matplotlib.pyplot as plt

class OrientationDetector:
    @staticmethod
    def rotate(image):
        
        # rotatedImage = OrientationDetector.boundingBoxRotate(image)
        
        # if rotatedImage is not None:
        #     return rotatedImage
        
        rotatedImage = OrientationDetector.projectionRotation(image)
        
        return rotatedImage
    
    @staticmethod
    def boundingBoxRotate(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rectangles = [cv2.boundingRect(contour) for contour in contours]
        outer_boundary = cv2.minAreaRect(np.concatenate([np.array([[x,y],[x+w,y+h]]) for x,y,w,h in bounding_rectangles]))
        box = cv2.boxPoints(outer_boundary)
        box = np.int0(box)
        width = np.max(box[:,0])-np.min(box[:,0])
        height = np.max(box[:,1])-np.min(box[:,1])
        
        if width > 2 * height:
            horizontalHist = np.sum(image, axis=1)
            return OrientationDetector.rotate180Check(image,horizontalHist)
        elif height > 2 * width:
            return OrientationDetector.rotateImage(image,90)
        else:
            return None
    
    @staticmethod
    def projectionRotation(image):
        verticalHist = np.sum(image, axis=0)
        vPeaks, _ = find_peaks(verticalHist,height= np.max(verticalHist) * 0.5, distance=10)
    
        numVZerosBetweenPeaks = 0
        for i in range(len(vPeaks) - 1):
            num_zeros = np.count_nonzero(verticalHist[vPeaks[i]:vPeaks[i+1]] == 0)
            if num_zeros > 0:
                numVZerosBetweenPeaks+=1    

        horizontalHist = np.sum(image, axis=1)
        hPeaks, _ = find_peaks(horizontalHist,height= np.max(horizontalHist) * 0.5, distance=10)
        numHZerosBetweenPeaks = 0
        for i in range(len(hPeaks) - 1):
            num_zeros = np.count_nonzero(horizontalHist[hPeaks[i]:hPeaks[i+1]] == 0)
            if num_zeros > 0:
                numHZerosBetweenPeaks+=1   
        
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rectangles = [cv2.boundingRect(contour) for contour in contours]
        if(not (len(bounding_rectangles) == 0)):
            outer_boundary = cv2.minAreaRect(np.concatenate([np.array([[x,y],[x+w,y+h]]) for x,y,w,h in bounding_rectangles]))
            box = cv2.boxPoints(outer_boundary)
            box = np.int0(box)
            width = np.max(box[:,0])-np.min(box[:,0])
            height = np.max(box[:,1])-np.min(box[:,1])
        else:
            width = image.shape[0]
            height = image.shape[1]
        numHZerosBetweenPeaks *= width
        numVZerosBetweenPeaks *= height
        
        if numHZerosBetweenPeaks == 0 and numVZerosBetweenPeaks == 0:
            #rotate 45
            rotatedImage = OrientationDetector.rotateImage(image,45)
        elif numHZerosBetweenPeaks < numVZerosBetweenPeaks:
            #rotate 90
            rotatedImage = OrientationDetector.rotateImage(image,90)
        else:
            # rotate 0 or 180
            rotatedImage = OrientationDetector.rotate180Check(image,horizontalHist)
        
        return rotatedImage
    
    @staticmethod
    def rotate180Check (image, horizontalHist):
        highestPeak = np.argmax(horizontalHist)
        rotatedImage = image
        
        sumLeft = 0
        i = highestPeak - 1
        while i >= 0 and horizontalHist[i] != 0:
            sumLeft += horizontalHist[i]
            i -= 1

        sumRight = 0
        j = highestPeak + 1
        while j < len(horizontalHist) and horizontalHist[j] != 0:
            sumRight += horizontalHist[j]
            j += 1
        
        if sumRight > sumLeft:
            rotatedImage = OrientationDetector.rotateImage(image,180)
        return rotatedImage
    
    @staticmethod
    def rotateImage(image, angle):
        height, width = image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotatedImage = cv2.warpAffine(image, rotationMatrix, (width, height))
        return rotatedImage
