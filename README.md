# Object-detection-and-tracking-demo
This project just change a little bit code in ahmetozlu project, add the class id for final output, not just show the object_XX. 
## Environment
* cuda (choose version by you os and tensorflow version)
* cudnn (choose version by you os and tensorflow version)
* windows 10
## Python library
* tensorflow-gpu 1.14
* openCV
* labelImg
## File explain
* object_tracking.py: It has opencv show functiion
* detection_layer.py: Tensorflow object detection
* backone.py: tracking function
* model : train by faster_rcnn_inception_v2_pets.config
## How to work
* After building eniveronment, personal model, and fix two python file
* commend : python object_tracking.py
## Reference
* Tensorflow object detection environment build
  * EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
* Tracking API 
  * ahmetozlu/tensorflow_object_counting_api
* video
  * New Taipei City Road Monitor
