import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
from PreDefinedSquareImageNet_model21 import PreDefinedSquareImageNet
import SlidingWindowDetector as SWD 
import GetDataSet
import AlexNet


import AddedLoss
import time
from timer import * 
import numpy as np 


#USER OPTIONS

#root directory of all scenes
data_path = '/playpen/ammirato/Data/HalvedRohitData/'
#where to save the model ater training
load_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' + 
             'exploring_pytorch/saved_models/recorded_models/')
model_name = 'model_38_2_0.903919560562.p'

#which/how many classes to learn
batch_size = 1
#desired image size. HxWxC.Must be square => H=W
image_size = [224,224,3]
org_img_dims = [1920/2, 1080/2]
#whether to use the GPU
cuda = False 

preload_images = False
reload_train_test = True 
reassign_transforms = False 

load_model=False
use_pretrained_alexnet = True
#show loss every X iterations
show_loss_iter = 50
#test the model every X epochs
test_epochs = 1
#max(ish) number of test images to use
max_test_size = 30000





test_list=[
             'Home_006_1',
             'Home_008_1',
             'Home_002_1'
             ]   

test_set = GetDataSet.get_alexnet_AVD(data_path,test_list,
                                      preload=preload_images,
                                      detection=True)
model = AlexNet.AlexNet(28)
model.load_state_dict(torch.load(load_path+model_name))

if cuda:
    model.cuda()


image_size=[224,224,3]
#image transforms
#normalize image to be [-1,1]
norm_trans = AVD_transforms.NormalizeRange(0.0,1.0)
norm_trans2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#resize the images to be image_size
resize_trans = AVD_transforms.ResizeImage(image_size[0:2],'fill')
to_tensor_trans = AVD_transforms.ToTensor()
classifier_trans = AVD_transforms.Compose([resize_trans,
                                            to_tensor_trans,
                                            norm_trans,
                                            norm_trans2])

detector = SWD.SlidingWindowDetector(model,image_trans=classifier_trans)


for il in range(len(test_set)):

    img,labels = test_set[165]

    results = detector(img)

    breakp = 0
    if il >= 0:
        break

