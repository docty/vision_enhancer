import cv2
import numpy as np
import os

class Equalization:
    def __init__(self, image_path="sample.jpg"):
        self.image_path=image_path
        equImage = self.index()
        self.display_image("Equalizer Image", equImage)
        #self.display_image("Ordinary Threshold", threshold_img)

    def display_image(self, title, image):
        try:
            from google.colab.patches import cv2_imshow 
            cv2_imshow(image)
        except ImportError:
            cv2.imshow(title, image)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
        
    def index(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        equ = cv2.equalizeHist(img)
        
        return equ
