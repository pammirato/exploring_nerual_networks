import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#reisze images so largest side is X
#do not pad, make square 

desired_size = 80 
base_path = '/playpen/ammirato/Data/big_bird_patches'
save_path = '/playpen/ammirato/Data/big_bird_patches_80'
image_names = os.listdir(base_path)

os.mkdir(save_path)

for name in image_names:
    #load image
    img = cv2.imread(os.path.join(base_path,name))

    #resize image so longest side is 200
    large_side = np.max(img.shape)

    scale_factor = float(desired_size)/large_side

    resized_img = cv2.resize(img,(int(img.shape[1]*scale_factor),
                                  int(img.shape[0]*scale_factor)))

    cv2.imwrite(os.path.join(save_path,name),resized_img)  


