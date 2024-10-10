import torch
import numpy as np
import scipy.ndimage as ndimage
import time
import math
from models.ggcnn import GGCNN
import cv2
import tifffile as tf
from imageio import imread


class GraspDetectionGGCNN():

    def __init__(self):
        model_path = r"/home/farhan/Desktop/VBRM_project/grasp_model/trained_models/GGCNN/ggcnn_epoch_23_cornell"
        self.network = torch.load(model_path, map_location='cpu')
        self.device = torch.device("cpu")  
        self.network.to(self.device)

    def pre_process(self,depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
        imh, imw = depth.shape
    # Crop
        depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                               (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
    
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

    
    def predict(self,depth, crop_size, out_size, crop_y_offset):
        depth, depth_nan_mask = self.pre_process(depth, crop_size, out_size, True, crop_y_offset)
        # normalize
        depth = np.clip((depth - depth.mean()), -1, 1)
        depth = np.uint8(depth)
        tensor = torch.from_numpy(depth).float()
        
        tensor = torch.reshape(tensor, (1, 300, 300))
        
        pred_out = self.network(tensor)
        
        #pos, cos, sin, width
        pred_out= np.array([pred_out[0].detach().numpy(),
                            pred_out[1].detach().numpy(),
                            pred_out[2].detach().numpy(),
                            pred_out[3].detach().numpy()])
        return pred_out



    def get_grasp(self,rgb, depth, crop_size, out_size, crop_y_offset):
        pred_out=self.predict(depth, crop_size, out_size, crop_y_offset)
        imh, imw = depth.shape[:2]
        argmax = np.argmax(pred_out[0])
        x= argmax%300
        y= int(np.ceil(argmax/300))

        argmax = np.argmax(pred_out[0])
        width= pred_out[3][0][y][x]*150
        angle = 0.5 * np.arctan2(pred_out[2][0][y][x],pred_out[1][0][x][y]) 
        
        grasp = np.array([x,y,angle,width])

        self.draw(rgb,grasp,imw,imh,crop_y_offset)

        return grasp


    def draw(self,rgb, grasp,imw,imh,crop_y_offset):
        x=grasp[0]
        y=grasp[1]
        angle=grasp[2]
        width=grasp[3]

        x=x+imw*0.5-150
        y=y+imh*0.5-150-crop_y_offset
        p1_x= int(x+(width*0.5*np.cos(angle)))
        p2_x= int(x-(width*0.5*np.cos(angle)))

        p1_y= int(y+(width*0.5*np.sin(angle)))
        p2_y= int(y-(width*0.5*np.sin(angle)))
        print(p1_x, p1_y, p2_x, p2_y)

        test = rgb.copy()
        current_frame = cv2.cvtColor(test,cv2.COLOR_RGB2BGR)

        # object= cv2.imread("pcd0100r.png", cv2.IMREAD_COLOR)  
        grasp_gg= cv2.line(current_frame,(p1_x,p1_y),(p2_x,p2_y),(255,0,0),5)
        cv2.imshow("Grasp_GGCNN", grasp_gg)
        cv2.waitKey(0)

    def testing_model(self):

        depth = imread(r'/home/farhan/Desktop/VBRM_project/grasp_model/test_dataset_cornel/pcd0100d.tiff')

        imh, imw = depth.shape[:2]
        out_size=300
        crop_y_offset=40
        crop_size=300

        grasp= self.get_grasp(depth, crop_size, out_size, crop_y_offset)
        self.draw(grasp,imw,imh,crop_y_offset)




if __name__ == "__main__":
    grasp_model = GraspDetectionGGCNN()
    grasp_model.testing_model()
    
