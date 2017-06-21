import torch
import torch.utils.data
import torchvision.models as torch_models
import torchvision.transforms as torch_transforms
from torch.autograd import Variable
import torch.nn.functional as F

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
import GetDataSet
import DetecterEvaluater

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir

import numpy as np

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
         'Home_006_1',
         'Home_008_1',
         'Home_002_1'
         ]   

#CREATE TRAIN/TEST splits
test_set = GetDataSet.get_fasterRCNN_AVD(data_path,scene_list,
                                           preload=False)

#create train/test loaders, with CUSTOM COLLATE function
testloader = torch.utils.data.DataLoader(test_set,
                                          batch_size=1,
                                          shuffle=False,
                                          collate_fn=AVD.collate)


# load net
trained_model = '/playpen/ammirato/Documents/exploring_neural_networks/exploring_pytorch/saved_models/fasterRCNN_avd.h5'
net = FasterRCNN(classes=test_set.get_class_names(), debug=False)
network.load_net(trained_model, net)
print('load model successfully!')

net.cuda()
net.eval()



    
evaluater = DetecterEvaluater.DetectorEvaluater(detector, testloader,chosen_ids)


pr = evaluater.run()



