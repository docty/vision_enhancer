import cv2
import numpy as np
import os

class Clahe:
    def __init__(self, image_path="sample.jpg"):
        self.image_path=image_path
        threshold_img, clahe_img = self.index()
        self.display_image("CLAHE Image", clahe_img)

    def display_image(self, title, image):
        try:
            from google.colab.patches import cv2_imshow 
            cv2_imshow(image)
        except ImportError:
            cv2.imshow(title, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    def index(self):
        image = cv2.imread(self.image_path)
        image_resized = cv2.resize(image, (500, 600))
        image_bw = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5)
        clahe_img = np.clip(clahe.apply(image_bw) + 30, 0, 255).astype(np.uint8)
        _, threshold_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
        return threshold_img, clahe_img
