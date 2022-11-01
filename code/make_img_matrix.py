import torch
import os
import numpy as np
import cv2

'''
Concatenate images into a matrix for figures
'''

num_imgs = 8            # Total number of images to put in the matrix
img_dims = None
matrix_dims = (2, 4)    # Total dimension of the matrix

dir = os.path.join('data', 'eval_data', 'train')
fns = os.listdir(dir)

for i in range(min(num_imgs, len(fns))):
    img = cv2.imread(os.path.join(dir, fns[i]))
    img = img[100 : -100, :]


    cv2.imshow('Image', img)
    cv2.waitKey(0)
