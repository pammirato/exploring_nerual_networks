import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#crop and save X views of each bigbird object.
#no resizing/warping

d_path = '/playpen/ammirato/Data/'
bb_path =  d_path + 'BigBIRD/'
save_path = d_path + 'big_bird_patches/'

img_names = ['NP2_45', 'NP2_225']


bb_names = os.listdir(bb_path)


for name in bb_names:
    print name
    for img_name in img_names:
        img1 = cv2.imread(bb_path + name + '/rgb/' +  img_name + '.jpg')
        mask1 = cv2.imread(bb_path + name + '/masks/' + img_name + '_mask.pbm')

        rows,cols = np.where(mask1[:,:,0] < 1)
        
        ca = int((cols.max()-cols.min()) * .1)
        ra = int((rows.max()-rows.min()) * .1)

        if name == 'coca_cola_glass_bottle':
            rmin = rows.min()
            rows[-1] = rows.max()+130
            rows[0] = rmin

        if name == 'listerine_green' or name == 'softsoap_clear':
            rmin = rows.min()
            rows[-1] = rows.max()+40
            rows[0] = rmin



        crop = img1[rows.min()-ra:rows.max()+ra,
                    cols.min()-ca:cols.max()+ca,
                    :]

   
        cv2.imwrite(save_path + name + '_' + img_name + '.jpg', crop) 

    



#desired_size = 80 
#base_path = '/playpen/ammirato/Data/big_bird_crops'
#image_names = os.listdir(base_path)
#
#
#for name in image_names:
#    #load image
#    img = cv2.imread(os.path.join(base_path,name))
#
#    #resize image so longest side is 200
#    large_side = np.max(img.shape)
#
#    scale_factor = float(desired_size)/large_side
#
#    resized_img = cv2.resize(img,(int(img.shape[1]*scale_factor),
#                                  int(img.shape[0]*scale_factor)))
#    r_shape = resized_img.shape 
#    
#    blank_img = np.zeros((desired_size,desired_size,3))
#
#    blank_img[0:r_shape[0],0:r_shape[1],:] = resized_img
#
#    cv2.imwrite(os.path.join(base_path,name),blank_img)  


