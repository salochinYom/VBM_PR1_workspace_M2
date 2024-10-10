
import torch.nn.functional as F
import argparse
import logging
import time
import numpy as np
import torch.utils.data

from imageio import imread
import cv2
import torch
import matplotlib.pyplot as plt
import sys



sys.path.append('/home/farhan/vbrm_project/src/grasp_model/grasp_model')
import post_process
from grasp import detect_grasps

class GraspDetectionNet():
    
    def __init__(self):
        model_path = r"/home/farhan/Desktop/VBRM_project/grasp_model/trained_models/GRConvNet3/epoch_30_iou_0.97"    # loaded a pre-trained model 
        self.network = torch.load(model_path, map_location='cpu')
        self.device = torch.device("cpu")  
        self.network.to(self.device)


    def load_and_preprocess_image(self, rgb_img, depth_img):

        # og_rgb = rgb_img.copy() 
        og_depth = depth_img.copy()

        rgb_img_cropped,rgb_offset = self.crop_image(rgb_img)  
        depth_img_cropped ,depth_offset= self.crop_image(depth_img)

        # Depth image pre-processing
        depth_img_preprocessed = self.preprocess_depth_image(depth_img_cropped)

        return rgb_img_cropped, depth_img_preprocessed,rgb_offset,og_depth

    def crop_image(self, image, crop_size=(224, 224)):
    #     height, width = image.shape[:2]
    #     crop_width, crop_height = crop_size

    #     aspect_ratio = width / height
    #     if aspect_ratio > 1:  # Image is wider than it is tall
    #         new_width = crop_width
    #         new_height = int(crop_width / aspect_ratio)
    #     else:  # Image is taller than it is wide or square
    #         new_height = crop_height
    #         new_width = int(crop_height * aspect_ratio)

    # # Center cropping
    #     start_x = (width - new_width) // 2
        
    #     start_y = (height - new_height) // 2
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        crop_width, crop_height = crop_size

        start_x = center_x - crop_width // 2
        start_y = center_y - crop_height // 2

        return image[start_y:start_y + crop_height, start_x:start_x + crop_width] , (start_x, start_y)

    def preprocess_depth_image(self, depth_image):
        depth_copy = depth_image.copy()
        depth_copy = np.pad(depth_copy, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
        mask = np.isnan(depth_copy).astype(np.uint8)

        # Dilation to smoothen the mask
        mask_dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        # Replace NaNs with zeros
        depth_copy[mask_dilated == 1] = 0

        # Scale the depth image for normalization
        max_val = np.max(np.abs(depth_copy))
        depth_copy = depth_copy.astype(np.float32) / max_val
       
        # Inpainting to fix missing pixels
        depth_inpainted = cv2.inpaint(depth_copy, mask_dilated, 1, cv2.INPAINT_NS)

        return depth_inpainted[1:-1, 1:-1]
        
        # return cv2.resize(depth_inpainted[1:-1, 1:-1], (224, 224))


    def format_inputs(self, rgb_img, depth_img):
        # Normalize RGB Image
        rgb_img = rgb_img.astype(np.float32) / 255.0
        rgb_img -= rgb_img.mean()  
        rgb_img = rgb_img.transpose((2, 0, 1)) 

        # Normalize Depth Image
        depth_img = np.clip(depth_img - np.mean(depth_img), -1, 1)
        depth_img = np.uint8(depth_img)


        # Concatenate RGB and depth
        combined_input = np.vstack([np.expand_dims(depth_img, 0), rgb_img])

        return torch.from_numpy(combined_input.astype(np.float32)).to(self.device)

    def predict_grasp(self, rgb_img, depth_img):
        input_tensor = self.format_inputs(rgb_img, depth_img).unsqueeze(0)

        with torch.no_grad():
            output = self.network.predict(input_tensor)

            quality_img = output['pos']   # model returns 4 outputs 
            cos_angle = output['cos']
            sin_angle = output['sin']
            width_img = output['width']
        # Post-process output
        return post_process.post_process_output(quality_img, cos_angle, sin_angle, width_img)

    def visualize_grasp(self, rgb_img, grasp_rectangles,offset):
            # fig, ax = plt.subplots()
            # ax.imshow(rgb_img)
            # for grasp in grasp_rectangles:
            #     grasp.plot(ax)
            # ax.set_title('Predicted Grasp Rectangle')
            # ax.axis('off')
            # plt.savefig('grasp_prediction.png')
            # plt.show()

        fig, ax = plt.subplots()
        ax.imshow(rgb_img)  # Show the original image

        # Adjust grasp rectangle coordinates by the crop offset
        for grasp in grasp_rectangles:
            # Adjust the center of the grasp rectangle by adding the crop offset
            # adjusted_center = (grasp.center[0] + offset[0], grasp.center[1] + offset[1])
            adjusted_center = (grasp.center[0] , grasp.center[1])
            grasp.center = adjusted_center
            
            print(f"Adjusted Grasp Center: {adjusted_center}") 
            

            grasp.plot(ax)

        ax.set_title('Predicted Grasp Rectangle')
        ax.axis('off')
        plt.savefig('grasp_prediction.png')
        plt.show()


    def run_grasp_detection(self, rgb, depth):
        rgb_img, depth_img,offset,og_depth  = self.load_and_preprocess_image(rgb,depth)
        quality, angle, width  = self.predict_grasp(rgb_img, depth_img)

        grasp_rectangles = detect_grasps(quality, angle, width, no_grasps=1) 

        x_cen = round(grasp_rectangles[0].center[0], 5)
        y_cen = round(grasp_rectangles[0].center[1] , 5)

        # x_cen = round(grasp_rectangles[0].center[0] + offset[0], 5)
        # y_cen = round(grasp_rectangles[0].center[1] + offset[1], 5)
        w = round(grasp_rectangles[0].width, 5)
        h = round(grasp_rectangles[0].length, 5)
        theta = round(grasp_rectangles[0].angle, 5)

        print(f"Grasp Center: ({x_cen}, {y_cen}), Width: {w}, Height: {h}, Angle: {theta}")
        self.visualize_grasp(rgb_img, grasp_rectangles,offset)
        return [x_cen, y_cen, w, h, theta], og_depth, grasp_rectangles
    

    def testing_model(self):
        rgb_img = imread(r"/home/farhan/Desktop/VBRM_project/grasp_model/test_dataset_cornel/pcd0100r.png")
        depth_img = imread(r"/home/farhan/Desktop/VBRM_project/grasp_model/test_dataset_cornel/pcd0100d.tiff")

        original_rgb = rgb_img.copy()
        print(original_rgb.shape)

        _,_,grasp = self.run_grasp_detection(rgb_img,depth_img)


if __name__ == "__main__":
    grasp_model = GraspDetectionNet()
    grasp_model.testing_model()
