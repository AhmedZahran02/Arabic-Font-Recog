import cv2
import numpy as np
from skimage.morphology import skeletonize

class Preprocessing:
    @staticmethod
    def crop_to_fit_white_with_border(image, border_size=10):

        _, binary = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x, y, w, h = cv2.boundingRect(contours[0])

        x -= border_size
        y -= border_size
        w += 2 * border_size
        h += 2 * border_size

        x = max(x, 0)
        y = max(y, 0)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        cropped_image = image[y:y+h, x:x+w]
        
        return cropped_image
    
    @staticmethod
    def calculate_entropy(image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        non_zero_indices = np.where(hist > 0)
        entropy = -np.sum(hist[non_zero_indices] * np.log2(hist[non_zero_indices]))
        return entropy

    @staticmethod
    def detect_salt_and_pepper_noise(image, entropy_threshold=7.0):
        entropy = Preprocessing.calculate_entropy(image)
        if entropy > entropy_threshold:
            return True
        else:
            return False
        
    @staticmethod
    def skeletonize(binary_image):
        skeleton = np.zeros(binary_image.shape, dtype=np.uint8)
        eroded = np.zeros(binary_image.shape, dtype=np.uint8)
        temp = np.zeros(binary_image.shape, dtype=np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

        while True:
            cv2.erode(binary_image, element, eroded)
            cv2.dilate(eroded, element, skeleton)
            cv2.subtract(binary_image, skeleton, temp)
            cv2.bitwise_or(skeleton, temp, skeleton)
            binary_image = eroded.copy() 
            if cv2.countNonZero(binary_image) == 0:
                break

        return skeleton