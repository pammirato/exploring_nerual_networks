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

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
             'Home_008_1',
            # 'Home_002_1'
             ]

test_set = GetDataSet.get_alexnet_AVD(data_path,scene_list)

testloader = torch.utils.data.DataLoader(test_set,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4, 
                                         collate_fn=AVD.collate)



#Define model, optimizer, and loss function
if use_alexnet_model:
    model = AlexNet.AlexNet(28)
    model.load_state_dict(torch.load(load_path+model_name))

else:#use predefined model
    model = PreDefinedSquareImageNet(image_size,num_classes)
    model.load_state_dict(torch.load(load_path+load_name))


num_correct = 0
num_total = 0
acc_by_class = np.zeros((num_classes,3))
thresholds = np.linspace(0,.9,10)
acc_by_threshold = np.zeros((len(thresholds),2))

#VIS LOOP
for il,data in enumerate(testloader):
  
    if il%50 == 0:
        print '{}/{}'. format(il,len(test_set))
 
    #get the images and labels for this batch 
    batch_imgs,batch_labels = data
   
    #get prediction and score 
    pred = model(Variable(batch_imgs))
    pred_score, pred_class = torch.nn.functional.softmax(pred).topk(1)
    
    #record result 
    #if the prediciton is correct
    if batch_labels[0] == pred_class.data.numpy()[0,0]:
        num_correct += 1
        acc_by_class[batch_labels[0],0] +=1

        
        for kl in range(len(thresholds)):
            if pred_score.data.numpy()[0,0] > thresholds[kl]:
               acc_by_threshold[kl,0] +=1 

    num_total += 1
    acc_by_class[batch_labels[0],1] +=1







for il in range(acc_by_class.shape[0]):
    acc_by_class[il,2] = 100*(acc_by_class[il,0]/acc_by_class[il,1])

for il in range(acc_by_threshold.shape[0]):
    acc_by_threshold[il,1] = 100.0*(acc_by_threshold[il,0] / float(num_total))

print 'Correct: {}  Total {}  Percent: {}'.format(num_correct, num_total, 
                                                 num_correct/float(num_total)) 

print acc_by_class


#plot acc by class
plt.figure(1)
#plt.bar(range(len(acc_by_class)),[x for _,_,x in acc_by_class.values()])
plt.bar(range(len(acc_by_class)),[x for _,_,x in acc_by_class])
#plt.xticks(range(len(acc_by_class)),acc_by_class.keys())
#plt.xticks(range(len(acc_by_class)),range(len(acc_by_class)))
plt.title('Histogram of Accuracies by Class')
plt.xlabel('class id')
plt.ylabel('Accuracy')
plt.draw()
plt.pause(.001)




#plot acc by thresh 
plt.figure(2)
plt.bar(range(len(acc_by_threshold)),[x for _,x in acc_by_threshold])
#plt.xticks(range(len(acc_by_class)),acc_by_class.keys())
#plt.xticks(range(len(acc_by_class)),range(len(acc_by_class)))
plt.title('Histogram of Accuracies by Threshold')
plt.xlabel('Score threshold*10')
plt.ylabel('Accuracy')
plt.draw()
plt.pause(.001)






