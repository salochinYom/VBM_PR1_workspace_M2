#stupid absolute path imports for ggcnn
import sys
import os
#gets the absolute path to the directory that contains the useful stuff to make the things
absPath = os.path.dirname(os.path.realpath(__file__))
#adds said path to the thing
sys.path.append(absPath)


import torch
import utils
from models.ggcnn import GGCNN
import numpy as np
import scipy.ndimage as ndimage
import time
import math

import cv2
import tifffile as tf

def pre_process(depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
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

    
def predict(depth, crop_size, out_size, crop_y_offset):
    depth, depth_nan_mask = pre_process(depth, crop_size, out_size, True, crop_y_offset)
    # normalize
    depth = np.clip((depth - depth.mean()), -1, 1)
    tensor = torch.from_numpy(depth).float()
    
    tensor = torch.reshape(tensor, (1, 300, 300))
    
    pred_out = model(tensor)
    
    #pos, cos, sin, width
    pred_out= np.array([pred_out[0].detach().numpy(),
                        pred_out[1].detach().numpy(),
                        pred_out[2].detach().numpy(),
                        pred_out[3].detach().numpy()])
    return pred_out



def get_grasp(depth, crop_size, out_size, crop_y_offset):
    pred_out=predict(depth, crop_size, out_size, crop_y_offset)

    argmax = np.argmax(pred_out[0])
    x= argmax%300
    y= int(np.ceil(argmax/300))

    argmax = np.argmax(pred_out[0])
    width= pred_out[3][0][y][x]*150
    angle = 0.5 * np.arctan2(pred_out[2][0][y][x],pred_out[1][0][x][y]) 
    
    return np.array([x,y,angle,width])


def draw(grasp):
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

    object= cv2.imread(absPath + "/pcd0100r.png", cv2.IMREAD_COLOR)
    object= cv2.line(object,(p1_x,p1_y),(p2_x,p2_y),(255,0,0),5)
    cv2.imshow("Grasp", object)
    cv2.waitKey(0)







if __name__ == '__main__':
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model = torch.load(absPath + '/ggcnn_weights_cornell/ggcnn_epoch_23_cornell', map_location=torch.device(device), weights_only=False)
    print(model)

    depth = cv2.imread(absPath + '/pcd0100d.tiff', -1)
    print(depth)
    imh, imw = depth.shape
    print(imh, " ", imw)
    out_size=300
    crop_y_offset=40
    crop_size=300

    grasp= get_grasp(depth, crop_size, out_size, crop_y_offset)
    draw(grasp)