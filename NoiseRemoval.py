import cv2
from skimage.restoration import denoise_nl_means
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

class NoiseRemoval:
    @staticmethod
    def applyGaussianBlur(image, kernel_size=(5, 5), sigma_x=0):
        try:
            blurred_image = cv2.GaussianBlur(image, kernel_size, sigma_x)
            return blurred_image
        except Exception as e:
            print(f"Error applying Gaussian blur: {e}")
            return None
    
    @staticmethod
    def sharpenImage(image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpImage = cv2.filter2D(image, -1, kernel)
        return sharpImage
    
    @staticmethod
    def applyMedianFilter(image, kernel_size=5):
        filteredImage = cv2.medianBlur(image, kernel_size)
        return filteredImage
