######## Video Object Detection and Tracking #########
#
# Author: Teng Yang Yu
# Date: 2020/07/27
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# It adds tracking function
# frame of the video.

## The main framwork read from 
## https://github.com/ahmetozlu/tensorflow_object_counting_api

import numpy as np
import cv2
import backbone
import imutils
from utils.object_tracking_module import tracking_layer
                     
cap = cv2.VideoCapture("video/gar1.AVI")


while(True):            
    ret, img = cap.read()
    if img is None:
    	break           
    np.asarray(img)
    processed_img = backbone.processor(img)
    image = imutils.resize(processed_img, width=1000)
    cv2.imshow('object tracking',image)


    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
        
print("end of the video!")
# Clean up
cap.release()
cv2.destroyAllWindows()