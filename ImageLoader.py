import cv2
import matplotlib.pyplot as plt
import glob

class ImageLoader:
    
    @staticmethod
    def loadImage(path, filename):
        fullPath = path + filename
        try:
            grayImageRead = cv2.imread(fullPath,0)
            if grayImageRead is None:
                raise FileNotFoundError(f"Image not found: {fullPath}")
            return grayImageRead
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def loadImages(path):
        path += "*"
        try:
            images = []
            for file in glob.glob(path):
                grayImageRead = cv2.imread(file,0)
                if grayImageRead is None:
                    raise FileNotFoundError(f"Image not found: {file}")
                images.append(grayImageRead)
            return images
        except Exception as e:
            print(f"Error loading images: {e}")
            return None
        
    @staticmethod
    def print(images):
        if images is not None:
            if isinstance(images, list):
                # If there are multiple images, display them in subplots
                for i, image in enumerate(images):
                    plt.imshow(image,cmap='gray')
                    plt.show()
            else:
                # If there's only one image, display it without subplots
                plt.imshow(images,cmap='gray')
                plt.show()
        else:
            print("Error printing images.")
            