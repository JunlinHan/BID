import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random

# random crop the alpha map to fit the shape of image
def get_random_crop(texture, alpha, crop_height):

    max_y = texture.shape[0] - crop_height
    y = np.random.randint(0, max_y)
    crop_texture = texture[y: y + crop_height, :, :]
    crop_alpha = alpha[y: y + crop_height, :, :]

    return crop_texture,crop_alpha

# 2048 * 1024
def get_position_matrix(texture,alpha,output_size,img):
    h,w = output_size

    texture_size = texture.shape[0]
    factor = h/w

    crop_w = texture_size
    crop_h = int(crop_w*factor)

    texture,alpha = get_random_crop(texture,alpha, crop_h)

    texture = cv2.resize(texture,(output_size[1],output_size[0]))
    alpha = cv2.resize(alpha, (output_size[1],output_size[0]))
    alpha = cv2.blur(alpha,(5,5))

    position_matrix = np.mgrid[0:h,0:w]

    position_matrix[0,:,:] = position_matrix[0,:,:] + texture[:,:,2]*(texture[:,:,0]/255)
    position_matrix[1,:, :] = position_matrix[1,:, :] + texture[:, :, 1]*(texture[:,:,0]/255)
    position_matrix = position_matrix*(alpha[:,:,0]>255*0.3)


    return position_matrix,alpha




