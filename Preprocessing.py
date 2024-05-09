import cv2

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