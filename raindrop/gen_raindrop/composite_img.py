import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from get_position_matrix import get_position_matrix


def composition_img(img,alpha,position_matrix,length=2):
    h, w = img.shape[0:2]
    dis_img = img.copy()

    for x in range(h):
        for y in range(w):
            u,v = int(position_matrix[0,x,y]/length),int(position_matrix[1,x,y]/length)
            if (u != 0 and v != 0):
                if((u<h) and (v<w)):
                    dis_img [x,y,:] = dis_img[u,v,:]
                elif(u<h):
                    print(w)
                    dis_img[x, y, :] = dis_img[u, np.random.randint(0,w-1), :]
                elif(v<w):
                    print(v)
                    dis_img[x, y, :] = dis_img[np.random.randint(0,h-1), v, :]

    dis_img = cv2.blur(dis_img,(3,3))*(0.9)


    img = (alpha/255)*dis_img + (1-(alpha/255))*img
    img = np.array(img,dtype=np.uint8)
    return img

imgs = os.listdir('../boon2')
alpha_imgs = os.listdir('../gen_raindrop/alpha_textures/alpha/')
print(alpha_imgs)

for i in range(50):
    print(i)
    alpha_img_name = alpha_imgs[random.randint(0,len(alpha_imgs)-1)]
    texture_img_name = 'texture_'+alpha_img_name

    img = cv2.imread(os.path.join('../boon2/',imgs[i]))
    alpha_img = cv2.imread(os.path.join('../gen_raindrop/alpha_textures/alpha/',alpha_img_name))
    texture_img = cv2.imread(os.path.join('../gen_raindrop/alpha_textures/texture/',texture_img_name))

    position_matrix, alpha = get_position_matrix(texture_img,alpha_img,img.shape[0:2],img)
    img = composition_img(img,alpha,position_matrix)
    cv2.imwrite('../output/raindrop'+str(i)+'.png',img)