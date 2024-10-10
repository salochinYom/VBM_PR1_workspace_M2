#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

from std_msgs.msg import Float64MultiArray
from transforms3d import euler
from cv_bridge import CvBridge


from geometry_msgs.msg import PoseStamped  # From original repository - generate_grasp function - antipodal grasp grconv

class Generate3DGrasp(Node):
    def __init__(self):
        super().__init__('grasp_generator_3d')
        
  
        self.grasp_box = self.create_subscription(+
            Float64MultiArray,
            '/grasp_bounding_box',
            self.grasp_bounding_box_callback,
            10
        )
        self.dp_img = self.create_subscription(
            Image,
            # '/realsense/depth/image_raw',
            '/vbrm_project/depth_image',
            self.depth_image_callback,
            10
        )


        self.grasp_3d = self.create_publisher(PoseStamped, '/grasp_real_3d', 10)
        self.box = None
        self.dp_img_val = None
        self.camera_frame = 'lens'
        self.br = CvBridge()



    def grasp_bounding_box_callback(self,msg):

        self.box = msg.data
        self.get_logger().info(f"Grasp Box: {self.box}")
        self.generate_real3d_grasp()

    def depth_image_callback(self,msg):

        self.dp_img_val = self.br.imgmsg_to_cv2(msg)
        self.get_logger().info('Recieved -- now processing')

        self.generate_real3d_grasp()

    
    def generate_real3d_grasp(self):

        if self.box is None or self.dp_img_val is None:
            print("Grasp Box or depth image not found")
            return

        x_center,y_center,width,height,theta = self.box

        img_h , img_w = self.dp_img_val.shape


        self.focal_x = 554.256  # scaling factor for 2d image coordinate to 3d real world coordinates
        self.focal_y = 554.256
        self.cx = img_h//2   # camera frame principal point
        self.cy = img_w//2


        # Now we convert 2d pos to 3d values 

        real_depth = self.dp_img_val[int(y_center), int(x_center)]
        pos_x = (x_center - self.cx) * real_depth / self.focal_x
        pos_y = (y_center - self.cy) * real_depth / self.focal_y
        pos_z = 1.15 - real_depth   # getting height of object excluding enviroment factors

        real_pose = PoseStamped()
        real_pose.header.frame_id = self.camera_frame
        real_pose.header.stamp = self.get_clock().now().to_msg() 

        real_pose.pose.position.x = pos_x
        real_pose.pose.position.y = pos_y
        real_pose.pose.position.z = float(pos_z)

        # Calculate orientation
        angle = np.radians(theta)
        quat = euler.euler2quat(np.pi, 0, ((angle % np.pi) - np.pi/2))
        real_pose.pose.orientation.x = quat[1]
        real_pose.pose.orientation.y = quat[2]
        real_pose.pose.orientation.z = quat[3]
        real_pose.pose.orientation.w = quat[0]


        self.grasp_3d.publish(real_pose)
        self.get_logger().info(f"generated grasp pose in 3D: {real_pose.pose}")


def main(args=None):
    rclpy.init(args=args)
    grasp_pose_generator = Generate3DGrasp()
    rclpy.spin(grasp_pose_generator)
    grasp_pose_generator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        







