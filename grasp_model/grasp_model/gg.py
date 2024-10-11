
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
        model_path = r"/home/farhan/vbrm_project/src/grasp_model/grasp_model/trained_models/GRConvNet3/epoch_30_iou_0.97"  # loaded a pre-trained model 
        self.network = torch.load(model_path, map_location='cpu')
        self.device = torch.device("cpu")  
        self.network.to(self.device)

    def load_and_preprocess_image(self, rgb_img, depth_img):
        og_depth = depth_img.copy()
        depth_img_preprocessed = self.preprocess_depth_image(depth_img)
        return rgb_img, depth_img_preprocessed ,og_depth 

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

        return post_process.post_process_output(quality_img, cos_angle, sin_angle, width_img)

    def visualize_grasp(self, rgb_img, grasp_rectangles):
        fig, ax = plt.subplots()
        ax.imshow(rgb_img)  # Show the original image

        # Plot grasp rectangles without adjusting for any offset
        for grasp in grasp_rectangles:
            grasp.plot(ax)

        ax.set_title('Predicted Grasp Rectangle')
        ax.axis('off')
        plt.savefig('grasp_prediction.png')
        plt.show()
    
    def plot_metrics(self, quality, width, angle):
        # Create a grid of plots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot Quality
        axs[0].imshow(quality, cmap='RdYlBu_r')
        axs[0].set_title('Quality Map')
        axs[0].axis('off')

        # Plot Angle
        axs[2].imshow(angle, cmap='hsv_r')
        axs[2].set_title('Angle Map')
        axs[2].axis('off')

        # Plot Width
        axs[1].imshow(width, cmap='RdYlBu_r')
        axs[1].set_title('Width Map')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    def run_grasp_detection(self, rgb, depth):
        rgb_img, depth_img,og_depth  = self.load_and_preprocess_image(rgb,depth)
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
        self.visualize_grasp(rgb_img, grasp_rectangles)
        return [x_cen, y_cen, w, h, theta], og_depth, grasp_rectangles
    
    
    def testing_model(self):
        rgb_img = imread(r"/home/raval/robotic-grasping/cornell-dataset/01/pcd0138r.png")
        depth_img = imread(r"/home/raval/robotic-grasping/cornell-dataset/01/pcd0138d.tiff")
        original_rgb = rgb_img.copy()

        quality, width, angle = self.run_grasp_detection(rgb_img, depth_img)

        # Plot the metrics after grasp detection
        self.plot_metrics(quality, width, angle)


if __name__ == "__main__":
    grasp_model = GraspDetectionNet()
    grasp_model.testing_model()
