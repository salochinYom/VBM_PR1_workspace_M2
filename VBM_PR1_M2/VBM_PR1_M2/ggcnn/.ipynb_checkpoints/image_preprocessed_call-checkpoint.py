# 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

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

    
def post_process(points, angle, width_img, depth):
    imh, imw = depth.shape
    # x = ((np.vstack((np.linspace((imw - self.img_crop_size) // 2, (imw - self.img_crop_size) // 2 + self.img_crop_size, depth_crop.shape[1], np.float), )*depth_crop.shape[0]) - self.cam_K[0, 2])/self.cam_K[0, 0] * depth_crop).flatten()
    # y = ((np.vstack((np.linspace((imh - self.img_crop_size) // 2 - self.img_crop_y_offset, (imh - self.img_crop_size) // 2 + self.img_crop_size - self.img_crop_y_offset, depth_crop.shape[0], np.float), )*depth_crop.shape[1]).T - self.cam_K[1,2])/self.cam_K[1, 1] * depth_crop).flatten()
    # pos = np.dot(camera_rot, np.stack((x, y, depth_crop.flatten()))).T + np.array([[cam_p.x, cam_p.y, cam_p.z]])

    # width_m = width_img / 300.0 * 2.0 * depth_crop * np.tan(self.cam_fov * self.img_crop_size/depth.shape[0] / 2.0 / 180.0 * np.pi)

    best_g = np.argmax(points)

    best_position_x = points[best_g, 0]
    best_position_y = points[best_g, 1]
    best_position_z = points[best_g, 2]
    best_orientation = tfh.list_to_quaternion(tft.quaternion_from_euler(np.pi, 0, ((angle[best_g_unr]%np.pi) - np.pi/2)))
    # best_width = width_m[best_g_unr]

    return (best_position_x, best_position_y, best_position_z, best_orientation)





if __name__ == '__main__':
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model = torch.load('ggcnn_weights_cornell/ggcnn_epoch_23_cornell', map_location=torch.device(device), weights_only=False)
    print(model)

    # model.eval()
    
    #SET
    depth = cv2.imread('pcd0100d.tiff', -1)
    imh, imw = depth.shape

    out_size=300
    crop_y_offset=40
    crop_size=300

    #PREDICT
    depth, depth_nan_mask = pre_process(depth, crop_size, out_size, True, crop_y_offset=crop_y_offset)
    depth = np.clip((depth - depth.mean()), -1, 1)
    tensor = torch.from_numpy(depth).float()
    # print(tensor.size())
    tensor = torch.reshape(tensor, (1, 300, 300))
    print(tensor.size())
    #count run time
    before = time.time()
    #evaluate the ML thing
    pred_out = model(tensor)
    # print(len(pred_out))
    # pred_out=pred_out.detach().numpy()
    pred_out= np.array([pred_out[0].detach().numpy(),
                        pred_out[1].detach().numpy(),
                        pred_out[2].detach().numpy(),
                        pred_out[3].detach().numpy()])

    points_out = pred_out[0].squeeze()
    points_out[depth_nan_mask] = False

    # Calculate the angle map.
    cos_out = pred_out[1].squeeze()
    sin_out = pred_out[2].squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0

    width_out = pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1

    filters=(2.0, 1.0, 1.0)
    # Filter the outputs.
    if filters[0]:
        points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    if filters[1]:
        ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

    points_out = np.clip(points_out, 0.0, 1.0-1e-3)
    # return points_out, ang_out, width_out, depth
    
    print(post_process(points_out, ang_out, width_out, depth))

    # #stare at outputs
    # print(output)
    # #print(str(len(output)))
    # print(str(output[0].shape)) #grasp quality
    # print(str(output[1].shape)) #angle cos
    # print(str(output[2].shape)) #angle sin 
    # print(str(output[3].shape)) #grasp width
    # #display total run time
    # print(time.time()-before)

