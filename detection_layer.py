######## Video Object Detection and Tracking #########
#
# Author: Teng Yang Yu
# Date: 2020/07/27
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It add class id 

#------The main code get from--------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 14th August 2019
#----------------------------------------------

import numpy as np
import tensorflow as tf
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob
cwd = os.path.dirname(os.path.realpath(__file__))

from utils import visualization_utils
from utils import label_map_util


class ObjectDetector(object):
    def __init__(self):

        self.object_boxes = []
        os.chdir(cwd)
        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        detect_model_name = 'inference_graph_garbage'
        
        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'       
        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(CWD_PATH,'training','garbage_labelmap.pbtxt')
        
        self.detection_graph = tf.Graph()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')               
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')    

    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)       

    def box_normal_to_pixel(self, box, dim, class_id):    
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width), class_id]
        return np.array(box_pixel)       
        
    def get_localization(self, image, visual=False):         
        # Number of classes the object detector can identify
        NUM_CLASSES = 2
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)  
        
        
        with self.detection_graph.as_default():

              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})          
              if visual == True:
                  visualization_utils.visualize_boxes_and_labels_on_image_array_tracker(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.4,
                      skip_labels=True,
                      line_thickness=3)   
                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()               
              

              boxes=np.squeeze(boxes)
              classes =np.squeeze(classes)
              scores = np.squeeze(scores)  
              cls = classes.tolist()
              idx_vec = [i for i, v in enumerate(cls) if ((scores[i]>0.6))]
              # 有變化時會從[0]-->[0,1]
              print('show idex_vec::::',idx_vec)              
              if len(idx_vec) ==0:
                  print('there are not any detections, passing to the next frame...')
              else:
                  tmp_object_boxes=[]
                  for idx in idx_vec:
                      dim = image.shape[0:2]
                      class_id = classes[idx]                      
                      # print(class_id)
                      box = self.box_normal_to_pixel(boxes[idx], dim, class_id)
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                      ratio = box_h/(box_w + 0.01)
                      
                      #if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                      tmp_object_boxes.append(box)
                      #print(box, ', confidence: ', scores[idx], 'ratio:', ratio)                                                   
                  
                  self.object_boxes = tmp_object_boxes             
        return self.object_boxes                             
