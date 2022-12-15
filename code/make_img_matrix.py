import torch
import random
import os
import numpy as np
import cv2

'''
Concatenate images into a matrix for figures
'''

num_imgs = 3                # Total number of images to put in the matrix
img_dims = (440, 420, 3)    # Dimension of each image
matrix_dims = (1, 3)        # Total dimension of the matrix
img_matrix = np.zeros((img_dims[0] * matrix_dims[0], img_dims[1] * matrix_dims[1], 3), dtype = np.uint8)

dir = os.path.join('data', 'captions', 'ood_cap')
fns = os.listdir(dir)
fns = random.sample(fns, num_imgs)
fns = ['img_4.jpg', 'img_13.jpg', 'img_30.jpg']

for i in range(len(fns)):
    img = cv2.imread(os.path.join(dir, fns[i]))
    img = img[:-40, 110 : -110, :]

    img_idx = np.unravel_index(i, matrix_dims)
    img_matrix[img_idx[0] * img_dims[0] : (img_idx[0] + 1) * img_dims[0],
               img_idx[1] * img_dims[1] : (img_idx[1] + 1) * img_dims[1], :] = img

cv2.imwrite(os.path.join(dir, 'img_matrix_ood.jpg'), img_matrix)
