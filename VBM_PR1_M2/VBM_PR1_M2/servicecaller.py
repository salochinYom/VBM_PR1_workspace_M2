import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from define_service.srv import ProcessImages
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
import numpy as np

class ServiceCaller(Node):
    def __init__(self):
        super().__init__('service_caller')
        self.srv=self.create_service(ProcessImages, 'process_images', self.process_images_callback)

    def process_images_callback(self, request, response):
        self.get_logger().info('Request received')
        depth_array = np.array(request.depth_image.data, dtype=float).flatten()
        result_msg = Float64MultiArray()
        result_msg.data = depth_array.tolist()
        response.result = result_msg        
        self.get_logger().info('Response sent')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ServiceCaller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
