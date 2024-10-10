#stupid absolute path imports for ggcnn
import sys
import os
#gets the absolute path to the directory that contains the useful stuff to make the things
absPath = os.path.dirname(os.path.realpath(__file__))
#adds said path to the thing
sys.path.append(absPath)

#is not a ros script 
import torch
import utils
from models.ggcnn import GGCNN
import numpy as np
import time

import cv2
import tifffile as tf


if __name__ == '__main__':
    #determines if we can use a GPU or not
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(device)
    model = torch.load(absPath + '/ggcnn_weights_cornell/ggcnn_epoch_23_cornell', map_location=torch.device(device), weights_only=False)
    #model(tensor) #input tensor then output tensor
    #model.eval()
    #print("hello world")
    print(model)
    #set model to evaluation mode
    model.eval()

    #import the image
    depth = cv2.imread(absPath + '/Jacquard_Samples/Samples/1a9fa4c269cfcc1b738e43095496b061/3_1a9fa4c269cfcc1b738e43095496b061_perfect_depth.tiff', -1)
    print(depth)

    #reshape the image to be 300 by 300 pixels
    depth = cv2.resize(depth, (300,300))
    #print(depth)

    #show the image
    #cv2.imshow('image', np.toInt(depth))

    #variable = input('input something!: ')


    #convert image to a tensor
    tesnor = torch.from_numpy(depth).float()

    #reshape mode

    tesnor = torch.reshape(tesnor, (1, 300, 300))
    #print(tesnor.shape)

    #count run time
    before = time.time()
    #evaluate the ML thing
    output = model(tesnor)

    #stare at outputs
    print(output)
    #print(str(len(output)))
    print(str(output[0].shape)) #grasp quality
    print(str(output[1].shape)) #angle cos
    print(str(output[2].shape)) #angle sin 
    print(str(output[3].shape)) #grasp width
    #display total run time
    print(time.time()-before)

    #complete the pipeline of image processing
    #get sin and cos 2d matrices
    sinArray = output[2].detach().numpy()
    cosArray = output[1].detach().numpy()
    #print(str(cosArray))

    #divide the two arrays
    divArray = np.divide(sinArray, cosArray) * 0.5
    print(divArray)

    #determine wtf is going on with the POS matrix
    posArray = output[0].detach().numpy()
    print(posArray.min())
    #normalize the positivity array to be > 0
    normFactor = -1 * posArray.min()
    posArray = posArray + normFactor
    print(posArray)

    





    #calculate the angle
    #angle = 0.5 * np.atan2(sinArray, output[3])