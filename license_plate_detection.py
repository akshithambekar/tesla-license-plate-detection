from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])

def identify(img):
    results = reader.readtext(img)
    for (bounding_box, text, prob) in results:
        (top_left, top_right, bottom_left, bottom_right) = bounding_box
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        for (text, _, _) in results:
            print(text)
    
if __name__ == "__main__":
    image_path = 'licenseplate.jpg'
    img = cv2.imread(image_path)
    img = cv2.resize(img, (600, 300))
    identify(img)