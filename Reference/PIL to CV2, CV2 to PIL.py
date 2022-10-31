#reference: https://www.zinnunkebi.com/python-opencv-pil-convert/
import cv2
import numpy
from PIL import Image

def PIL2OpenCV(pil_image):
    # open image using PIL

    # use numpy to convert the pil_image into a numpy array
    numpy_image=numpy.array(pil_image)  

    # convert to a openCV2 image and convert from RGB to BGR format
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    #display image to GUI
    cv2.imshow("PIL2OpenCV",opencv_image)

    return opencv_image

def OpenCV2PIL(opencv_image):

    #display image to GUI
    cv2.imshow("PIL2OpenCV", opencv_image)

    # convert from BGR to RGB
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # convert from openCV2 to PIL
    pil_image=Image.fromarray(color_coverted)

    return pil_image