import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



desired_size = 80 
base_path = '/playpen/ammirato/Data/big_bird_crops'
image_names = os.listdir(base_path)


for name in image_names:
    #load image
    img = cv2.imread(os.path.join(base_path,name))

    #resize image so longest side is 200
    large_side = np.max(img.shape)

    scale_factor = float(desired_size)/large_side

    resized_img = cv2.resize(img,(int(img.shape[1]*scale_factor),
                                  int(img.shape[0]*scale_factor)))
    r_shape = resized_img.shape 
    
    blank_img = np.zeros((desired_size,desired_size,3))

    blank_img[0:r_shape[0],0:r_shape[1],:] = resized_img

    cv2.imwrite(os.path.join(base_path,name),blank_img)  


