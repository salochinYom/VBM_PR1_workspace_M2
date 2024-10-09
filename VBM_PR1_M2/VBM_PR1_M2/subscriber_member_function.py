import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from model_service_interface.srv import ProcessImages
from cv_bridge import CvBridge

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/realsense/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/realsense/depth/image_raw', self.depth_callback, 10)
        self.newImage = False
        self.newDepth = False
        self.runAnotherCall = True
        

        self.grasp_box_pub = self.create_publisher(Float64MultiArray, '/grasp_bounding_box', 10)  
        
        self.model_client = self.create_client(ProcessImages, 'process_images')
        
        
        self.rgb = None
        self.depth = None

        self.br = CvBridge()

   
    def rgb_callback(self, msg):
        if(not self.newImage):
            self.newImage = True
            self.get_logger().info('Subing: image')
            self.rgb = msg
            self.doSomething()
        

    def depth_callback(self, msg):
        if(not self.newDepth):
            self.newDepth = True
            self.get_logger().info('Subing: depth')
            self.depth = msg
            self.doSomething()
        
    
    def doSomething(self):
        if(self.newDepth and self.newImage and self.runAnotherCall):
            self.runAnotherCall=False
            #Call model service
            while not self.model_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.get_logger().info('service called')
            request = ProcessImages.Request()
            request.depth_image = self.depth
            request.rgb_image = self.rgb
            self.future=self.model_client.call_async(request)
            self.future.add_done_callback(self.future_callback)
            # rclpy.spin_until_future_complete(self, self.future)


            
            self.get_logger().info('Subing: Hi')

    def future_callback(self, future):
        self.get_logger().info('future_callback')
        self.runAnotherCall=True
        self.newDepth = False
        self.newImage = False
        print(future.result().result.data.__len__())
        self.doSomething()
        
    

def main(args=None):
    rclpy.init()
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
if __name__ == '__main__':
    main()
