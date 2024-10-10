#stupid absolute path imports for ggcnn
import sys
import os
#gets the absolute path to the directory that contains the useful stuff to make the things
absPath = os.path.dirname(os.path.realpath(__file__))
#adds said path to the thing
sys.path.append(absPath)

import torch
#import utils
from models.ggcnn import GGCNN
import numpy as np
import scipy.ndimage as ndimage
import time
import math

import cv2
import tifffile as tf

#imports ros2
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images

class GCCNN_Processor(Node):
  #global varible stuff
  out_size=300
  crop_y_offset=40
  crop_size=300
  """
  Create a class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('ggcnn_processor')
      
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.

    # Create the publisher. This publisher will publish an image of the identified poses
    self.publisher_ = self.create_publisher(Image, 'output_image', 10)

    #create service for handling service requests for the things

      
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

    #setup pytorch stuff
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    self.model = torch.load(absPath + '/ggcnn_weights_cornell/ggcnn_epoch_23_cornell', map_location=torch.device(device), weights_only=False)
    #print(self.model)
    #set the model to evaluation mode
    self.model.eval()

    #test pipeline
    depth = cv2.imread(absPath + '/pcd0100d.tiff', -1)
    imh, imw = depth.shape
    print(imh, " ", imw)

    grasp= self.get_grasp(depth, self.crop_size, self.out_size, self.crop_y_offset, imh, imw)
    print(grasp)

    #get grasp visualization
    RGB_image = cv2.imread(absPath + "/pcd0100r.png", cv2.IMREAD_COLOR)
    RGB_viz = self.draw(grasp, RGB_image)
    #print("its working maybe")

    #publish the depth image
    while True:
      self.publisher_.publish(self.br.cv2_to_imgmsg(RGB_viz, encoding="bgr8"))


  def pre_process(self, depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0, imh=480, imw=640):
    # Crop
    depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                               (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
    
    #print(depth_crop)

    #Inpainting
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    kernel = np.ones((3, 3),np.uint8)
    depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)

    depth_crop[depth_nan_mask==1] = 0

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32) / depth_scale 

    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale

    depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:
        depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
        depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask
    else:
        return depth_crop
    
  def predict(self, depth, crop_size, out_size, crop_y_offset, imh, imw):
    depth, depth_nan_mask = self.pre_process(depth, crop_size, out_size, True, crop_y_offset, imh, imw)
    # normalize
    depth = np.clip((depth - depth.mean()), -1, 1)
    tensor = torch.from_numpy(depth).float()
    
    tensor = torch.reshape(tensor, (1, 300, 300))

    #run the pytorch model
    pred_out = self.model(tensor)
    
    #pos, cos, sin, width
    pred_out= np.array([pred_out[0].detach().numpy(),
                        pred_out[1].detach().numpy(),
                        pred_out[2].detach().numpy(),
                        pred_out[3].detach().numpy()])
    return pred_out
  
  def get_grasp(self, depth, crop_size, out_size, crop_y_offset, imh, imw):
    pred_out=self.predict(depth, crop_size, out_size, crop_y_offset, imh, imw)
    #print("hello world")

    argmax = np.argmax(pred_out[0])
    x= argmax%300
    y= int(np.ceil(argmax/300))

    argmax = np.argmax(pred_out[0])
    width= pred_out[3][0][y][x]*150
    angle = 0.5 * np.arctan2(pred_out[2][0][y][x],pred_out[1][0][x][y]) 
    
    return np.array([x,y,angle,width])
  
  #now returns image that can be output as a ros image
  def draw(self, grasp, RGB_image, imh= 480, imw=640):
    x=grasp[0]
    y=grasp[1]
    angle=grasp[2]
    width=grasp[3]

    x=x+imw*0.5-150
    y=y+imh*0.5-150-self.crop_y_offset
    p1_x= int(x+(width*0.5*np.cos(angle)))
    p2_x= int(x-(width*0.5*np.cos(angle)))

    p1_y= int(y+(width*0.5*np.sin(angle)))
    p2_y= int(y-(width*0.5*np.sin(angle)))
    print(p1_x, p1_y, p2_x, p2_y)

    #object= cv2.imread(absPath + "/pcd0100r.png", cv2.IMREAD_COLOR)
    object= cv2.line(RGB_image,(p1_x,p1_y),(p2_x,p2_y),(255,0,0),5)
    #cv2.imshow("Grasp", object)
    #cv2.waitKey(0)
    return object


def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  node = GCCNN_Processor()
  
  # Spin the node so the callback function is called.
  rclpy.spin(node)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  node.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()

if __name__ == '__main__':
    main()