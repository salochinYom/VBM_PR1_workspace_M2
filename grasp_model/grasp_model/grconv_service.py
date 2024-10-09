import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray 

from grconv_grasp import *
from ggcnn_grasp import *



class GraspService(Node):

    def __init__(self):

        super().__init__('Grasp_grconv_service')


        # subscribe to rgb and depth images from realsense topic
        self.rgb_img_subs = self.create_subscription(
            Image,
            '/realsense/image_raw', 
            self.rgb__callback,
            10)
        
        # Subscriber to depth image
        self.dp_img_sub = self.create_subscription(
            Image,
            '/realsense/depth/image_raw',
            self.depth_callback,
            10) 
        

        # publish grasp box

        self.grasp_box_pub = self.create_publisher(Float64MultiArray, '/grasp_bounding_box', 10)  
        # Lets define a service that takes  two strings for 2 models and run respective models

        # define service definition with name grconv.srv  - as a code for which model to use and the publish message type

        self.grasp_srv = self.create_service(grconv,'grconv_model',self.grasp_callback)

        self.model1 = GraspDetectionNet()
        self.model2 = GraspDetectionGGCNN()
        self.rgb = None
        self.depth = None

        self.br = CvBridge()



    def grasp_callback(self,request,response):

        if request.model == "use_grconvnet":
            grasp, depth,_ = self.model1.run_grasp_detection(self.rgb_image, self.depth_image)

            grasp_msg = Float64MultiArray()
            grasp_msg.data(grasp)
            

            self.get_logger().info('Generated Grasp using GRConvNet')
            response.grasp = grasp   # Assign the grasp message to the response
            
            self.grasp_box_pub.publish(grasp_msg)

        elif request.model == "use_ggcnn":
            grasp = self.model2.get_grasp(self.depth_image,300,300,40)

            grasp_msg = Float64MultiArray()
            grasp_msg.data(grasp)
            

            self.get_logger().info('Generated Grasp using GGCNN')
            response.grasp = grasp   # Assign the grasp message to the response
            
            self.grasp_box_pub.publish(grasp_msg)

        else:
            print("No input from User")

        return response
        
    def rgb_callback(self,msg):

        current_frame = self.br.imgmsg_to_cv2(msg)  # convert from ros message to cv2 object
        current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2RGB)   # converting from standard BGR to RGB image format for model processing

        self.rgb = current_frame

    
    def depth_callback(self,msg):

        depth_frame = self.br.imgmsg_to_cv2(msg)  # convert from ros message to cv2 object
        self.depth = depth_frame


def main(args=None):

    rclpy.init(args=args)
    gr_service = GraspService()

    rclpy.spin(gr_service)

    gr_service.destroy_node()

    rclpy.shutdown()
    

