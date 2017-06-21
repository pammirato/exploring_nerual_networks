import torch
import torch.utils.data
import torchvision.models as torch_models
import torchvision.transforms as torch_transforms
from torch.autograd import Variable
import torch.nn.functional as F

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
from PreDefinedSquareImageNet_model10 import PreDefinedSquareImageNet
import GetDataSet
import AlexNet

import numpy as np
import  SlidingWindowDetector as SWD
import DetecterEvaluater

#USER OPTIONS

#root directory of all scenes
data_path = '/playpen/ammirato/Data/HalvedRohitData/'
#where to save the model ater training
load_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' + 
             'exploring_pytorch/saved_models/recorded_models/') 
model_name = 'model_38_2_0.903919560562.p' 

use_alexnet_model = True

chosen_ids = range(28)
num_classes = len(chosen_ids) #4  
max_difficulty = 4
#standard CNN inputs

#standard CNN inputs
batch_size = 1 
#desired image size. HxWxC.Must be square => H=W
image_size = [224,224,3]
org_img_dims = [1920/2, 1080/2]

preload_images=False

#CREATE TRAIN/TEST splits



data_path = '/playpen/ammirato/Data/HalvedRohitData/'
scene_list=[
            # 'Home_006_1',
            # 'Home_008_1',
            'test',
            # 'Home_002_1'
             ]

test_set = GetDataSet.get_alexnet_AVD(data_path,scene_list,
                                        detection=True)
testloader = torch.utils.data.DataLoader(test_set,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=AVD.collate)



#Define model, optimizer, and loss function
if use_alexnet_model:
    model = AlexNet.AlexNet(28)
    model.load_state_dict(torch.load(load_path+model_name))

else:#use predefined model
    model = PreDefinedSquareImageNet(image_size,num_classes)
    model.load_state_dict(torch.load(load_path+load_name))



classification_trans = GetDataSet.get_alexnet_classification_transform()
detector = SWD.SlidingWindowDetector(model,
                               image_trans=classification_trans)

    
evaluater = DetecterEvaluater.DetectorEvaluater(detector, testloader,chosen_ids)


pr = evaluater.run()



