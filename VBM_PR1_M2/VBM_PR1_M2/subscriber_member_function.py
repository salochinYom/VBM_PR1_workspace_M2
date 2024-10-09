import random
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from define_service.srv import ProcessImages
from cv_bridge import CvBridge
from std_msgs.msg import Empty
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
import math


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/realsense/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/realsense/depth/image_raw', self.depth_callback, 10)
        self.newImage = False
        self.newDepth = False
        self.runAnotherCall = True

        # self.pause_phys = self.create_client(Empty, 'pause_physics')
        self.pause_phys = self.create_publisher(Empty, '/pause_physics', 10)

        self.grasp_box_pub = self.create_publisher(Float64MultiArray, '/grasp_bounding_box', 10)  
        
        self.model_client = self.create_client(ProcessImages, 'process_images')
        
        self.entity_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
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

            
            self.newDepth = False
            self.newImage = False
            
            # self.get_logger().info('Subing: Hi')

    def future_callback(self, future):
        self.get_logger().info('future_callback')
        self.runAnotherCall=True
        self.moveCam()
        
        print(future.result().result.data.__len__())
        self.doSomething()
        self.newDepth = False
        self.newImage = False
    
    def moveCam(self):
        # while not self.pause_phys.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for service...')
        # self.send_pause_request()
        self.pause_phys.publish(Empty())
        
        self.send_request()
        # x = random.random()*1.5-.75
        # y = random.random()*1.5-.75
        # z = random.random()*1.5-.75

        # state=EntityState()
        # state.name='/camera'
        # state.pose.position.x=x
        # state.pose.position.y=y
        # state.pose.position.z=z

        # while not self.entity_state_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for service...')
        # self.send_entity_state_request(state)



    # def send_pause_request(self):
    #     req = Empty.Request()
    #     self.future = self.pause_phys.call_async(req)
    #     rclpy.spin_until_future_complete(self, self.future)
    #     if self.future.result() is not None:
    #         self.get_logger().info('Physics paused successfully')
    #     else:
    #         self.get_logger().error('Failed to pause physics')

    def send_request(self):
        req = SetEntityState.Request()
        entity_state = EntityState()
        entity_state.name = 'camera'  # Use your camera entity's name
        entity_state.pose.position.x = self.randRange(-.25,.25)  # Desired X position
        entity_state.pose.position.y = self.randRange(-.25,.25)  # Desired Y position
        entity_state.pose.position.z = self.randRange(1,1.65)  # Desired Z position
        
        # Calculate the Euler angles to face the origin (0,0,0)
        x, y, z = entity_state.pose.position.x, entity_state.pose.position.y, entity_state.pose.position.z
        yaw = -math.atan2(y, x)#+self.randRange(-.05,.05)  # Add some noise to yaw
        hyp = math.sqrt(x**2 + y**2)
        pitch = math.atan2(z-.6, hyp)#+self.randRange(-.05,.05)  # Add some noise to pitch
        roll = 0.0  # Assuming no roll needed

        # Set orientation with calculated Euler angles
        quat = self.euler_to_quaternion(roll, pitch, yaw)
        entity_state.pose.orientation.x = quat[0]
        entity_state.pose.orientation.y = quat[1]
        entity_state.pose.orientation.z = quat[2]
        entity_state.pose.orientation.w = quat[3]
        entity_state.reference_frame = 'world'
        req.state = entity_state
        self.future = self.entity_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, self.future)
        if self.future.result() is not None:
            self.get_logger().info('Camera moved successfully')
        else:
            self.get_logger().error('Failed to move camera')

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    def randRange(self, min, max):
        return random.random()*(max-min)+min
        
        
    

def main(args=None):
    rclpy.init()
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
if __name__ == '__main__':
    main()
