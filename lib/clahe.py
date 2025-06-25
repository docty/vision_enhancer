import cv2
import numpy as np
import os

class Clahe:
    def __init__(self, image_path="sample.jpg", clipLimit=2.0, tileGridSize=(8, 8)):
        self.image_path = image_path
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def display_side_by_side(self, title, original, enhanced):
        # Resize to match dimensions if needed
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))

        # Add labels
        labeled_original = original.copy()
        labeled_enhanced = enhanced.copy()

        cv2.putText(labeled_original, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(labeled_enhanced, "Enhanced", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Combine and display
        combined = np.hstack((labeled_original, labeled_enhanced))
        try:
            from google.colab.patches import cv2_imshow
            cv2_imshow(combined)
        except ImportError:
            cv2.imshow(title, combined)
            cv2.waitKey(0)
             


    def index_1(self):
        image = cv2.imread(self.image_path)
        image_resized = cv2.resize(image, (500, 600))
        image_bw = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=self.clipLimit)
        clahe_apply = clahe.apply(image_bw)
        clahe_img = np.clip(clahe_apply + 30, 0, 255).astype(np.uint8)

        # Convert grayscale to BGR for side-by-side view
        original_display = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2BGR)
        enhanced_display = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

        self.display_side_by_side("CLAHE - index_1", original_display, enhanced_display)

    def index_2(self):
        image = cv2.imread(self.image_path)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(image_lab)

        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        l_channel_clahe = clahe.apply(l_channel)
        lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
        enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        self.display_side_by_side("CLAHE - index_2", image, enhanced_img)

    def index_3(self):
        image = cv2.imread(self.image_path)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v = hsv[:, :, 2]
        I = 255 - v
        clahe_I = clahe.apply(I)
        clahe_I_norm = clahe_I / 255
        gamma_5_I = np.power(clahe_I_norm, 5)

        gamma_5_I_top = np.clip(gamma_5_I * 255, 0, 255).astype('uint8')
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat_image = cv2.morphologyEx(gamma_5_I_top, cv2.MORPH_TOPHAT, structuring_element)
        tophat_image = tophat_image / 255

        tophat_image_flat = tophat_image.reshape(-1, 1)
        clahe_I_flat = clahe_I_norm.reshape(-1, 1)
        x = np.hstack((clahe_I_flat, tophat_image_flat))
        c = np.cov(x, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(c)
        max_index = np.argmax(eigenvalues)
        max_vector = eigenvectors[:, max_index]
        w1 = max_vector[0] / sum(max_vector)
        w2 = max_vector[1] / sum(max_vector)
        f = w1 * clahe_I_norm + w2 * tophat_image
        f = np.clip(f, 0, 1)
        f_inv = 1 - f
        hsv[:, :, 2] = (f_inv * 255).astype(np.uint8)
        enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.display_side_by_side("CLAHE - index_3", image, enhanced_img)
