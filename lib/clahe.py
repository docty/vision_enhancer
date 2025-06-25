import cv2
import numpy as np
import os

class Clahe:
    def __init__(self, image_path="sample.jpg"):
        self.image_path=image_path
         

    def display_image(self, title, image):
        try:
            from google.colab.patches import cv2_imshow 
            cv2_imshow(image)
        except ImportError:
            cv2.imshow(title, image)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
        
    def index_1(self):
        image = cv2.imread(self.image_path)
        image_resized = cv2.resize(image, (500, 600))
        image_bw = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=5)
        chahe_apply = clahe.apply(image_bw)

        clahe_img = np.clip(chahe_apply + 30, 0, 255).astype(np.uint8)

        _, threshold_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)

        self.display_image("CLAHE Image", clahe_img)
        #return threshold_img, clahe_img

        

    def index_2(self):
        image = cv2.imread(self.image_path)

        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(image_lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel_clahe = clahe.apply(l_channel)

        lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

        enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        self.display_image("CLAHE Image", enhanced_img)
        #return enhance_image
