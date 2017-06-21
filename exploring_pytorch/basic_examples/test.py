import torch
import torch.utils.data
import torchvision.models as torch_models
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
from PreDefinedSquareImageNet_model21 import PreDefinedSquareImageNet
import GetDataSet
import AlexNet
import SlidingWindowDetector as SWD


#TODO - make one 'ImageLoader' class that detection gui and main gui inheret




##################################################################
##################################################################
##################################################################


#USER INPUT
data_path = '/playpen/ammirato/Data/HalvedRohitData/'
scene_list=[
            # 'Home_006_1',
             'Home_008_1',
            # 'Home_002_1'
             ]
load_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' + 
             'exploring_pytorch/saved_models/recorded_models/')
model_name = 'model_38_2_0.903919560562.p'



dataset = GetDataSet.get_alexnet_AVD(data_path,scene_list)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         collate_fn=AVD.collate)

det_dataset = GetDataSet.get_alexnet_AVD(data_path,scene_list,detection=True)
det_dataloader = torch.utils.data.DataLoader(det_dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         collate_fn=AVD.collate)


