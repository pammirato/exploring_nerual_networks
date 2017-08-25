import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
import GetDataSet

from PreDefinedSquareImageNet_model21 import PreDefinedSquareImageNet
import AddedLoss
import time
from timer import * 
import numpy as np 

#USER OPTIONS

#root directory of all scenes
data_path = '/playpen/ammirato/Data/HalvedRohitData/'
#where to save the model ater training
save_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' + 
             'exploring_pytorch/saved_models/')
load_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' + 
             'exploring_pytorch/saved_models/recorded_models/')
save_name = 'model_41'
load_name = 'model_40_5_0.83148682618.p'
save_extension = '.p'
#which/how many classes to learn
chosen_ids = range(28) 
num_classes = len(chosen_ids) #4 
#standard CNN inputs
learning_rate = .00005
batch_size = 128 
max_epochs = 50  
use_class_weights = False 
use_added_loss = False 

#desired image size. HxWxC.Must be square => H=W
image_size = [224,224,3]
#whether to use the GPU
cuda = True 

preload_images = False 
reload_train_test = True

load_model = True
use_pretrained_alexnet = True
freeze_alexnet_conv_layers = False
#show loss every X iterations
show_loss_iter = 50
#test the model every X epochs
test_epochs = 1
#max(ish) number of test images to use
max_test_size = 30000


train_list=[
             'Home_001_1',
             'Home_001_2',
             'Home_003_1',
             'Home_003_2',
             'Home_004_1',
             'Home_004_2',
             'Home_005_1',
             'Home_005_2',
             'Home_014_1',
             'Home_014_2',
            ]
test_list=[
             'Home_006_1',
             'Home_008_1',
             'Home_002_1'
             ]


#CREATE TRAIN/TEST splits
if reload_train_test:
    train_set = GetDataSet.get_alexnet_AVD(data_path,train_list,
                                           preload=preload_images)
    test_set = GetDataSet.get_alexnet_AVD(data_path,test_list,
                                           preload=preload_images)

#create train/test loaders, with CUSTOM COLLATE function
trainloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          collate_fn=AVD.collate)
testloader = torch.utils.data.DataLoader(test_set,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn=AVD.collate)


#get class weights
#count_by_class = train_set.get_count_by_class(num_classes)
count_by_class = train_set.get_count_by_class()
total = len(train_set)
class_weights = np.zeros(num_classes)
for cid in count_by_class.keys():
    if count_by_class[cid] > 0:
        #TODO: make more robust. use class id, not index
        class_weights[cid] =  total / count_by_class[cid]
class_weights = torch.FloatTensor(class_weights)
if cuda:
    class_weights = class_weights.cuda()


#Define model, optimizer, and loss function
if use_pretrained_alexnet:
    model = models.alexnet(pretrained=True)
    #change ouput layer 
    model.classifier._modules['6'] = torch.nn.modules.linear.Linear(4096,num_classes) 

    if load_model:
        model.load_state_dict(torch.load(load_path + load_name))

    if freeze_alexnet_conv_layers:
        #freeze layers
        #as in http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward
        counter = 0
        params_to_optimize = []
        for param in model.parameters():
            if counter > 9:#11th and on are FC layers
                params_to_optimize.append(param)
                continue
            param.requires_grad = False
            counter +=1
        optimizer = torch.optim.Adam(params_to_optimize,lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
  
 
else:#make custom model
    model = PreDefinedSquareImageNet(image_size,num_classes)
    if load_model:
        model.load_state_dict(torch.load(load_path + load_name))
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    

#define loss function
if use_class_weights:
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
else:
    loss_fn = torch.nn.CrossEntropyLoss()
if use_added_loss:
    loss_fns = [torch.nn.CrossEntropyLoss(weight=class_weights),
                torch.nn.CrossEntropyLoss()]

#use gpu
if cuda:
    model.cuda()



#TRAIN LOOP
for epoch in range(max_epochs):
    for il,data in enumerate(trainloader):
      
        #get the images and labels for this batch 
        batch_imgs,batch_labels = data
        if cuda:
            batch_imgs,batch_labels = batch_imgs.cuda(),batch_labels.cuda() 

        #zero gradients, forward pass, compute loss
        optimizer.zero_grad()     
        y_pred = model(Variable(batch_imgs))
        if use_added_loss:
            losses = []
            for fn in loss_fns:
                losses.append(fn.forward(y_pred,Variable(batch_labels)))
    
            loss = sum(losses)
        else: 
            loss = loss_fn.forward(y_pred,Variable(batch_labels))

        #backward pass, update model parameters
        loss.backward()
        optimizer.step()
    
        #show loss every few iterations 
        if il % show_loss_iter == 0:
            print 'Iter: {}  Loss : {} '.format(il,loss.data[0])    

    #evaluate current model on a portion of the test set 
    if epoch % test_epochs == 0:
        num_correct = 0
        total = 0

        acc_by_class = np.zeros((num_classes,3))

        for il, data in enumerate(testloader):
            #get the data
            batch_imgs,batch_labels = data
            if cuda:
                batch_imgs = batch_imgs.cuda()
            #forward pass 
            y_pred = model(Variable(batch_imgs))
            #class predictions equal argmax of output layer
            #TODO: use tensor.topk
            pred_score,pred_class = torch.nn.functional.softmax(y_pred).topk(1)
            pred_score = pred_score.data.cpu()
            pred_class = pred_class.data.cpu()

            #see how many were wrong (non-zero is wrong)
            diff = batch_labels - pred_class
            num_correct += len(diff) - np.count_nonzero(diff.numpy())
            total += len(diff) 

            #keep track of num correcrt/total for each class
            for jl in range(batch_labels.size()[0]):
                gt = batch_labels[jl]
                pred = pred_class[jl]
                acc_by_class[gt,1] +=1
                if(gt == pred[0]):
                    acc_by_class[gt,0] +=1               
 

            #keep test size small for speed
            if total > max_test_size: 
                break
        acc = num_correct/float(total)
        print 'Test: Correct: {}  Total: {}  Accuracy: {}'.format(
                                                num_correct,
                                                total,
                                                acc)

        for il in range(acc_by_class.shape[0]):
            acc_by_class[il,2] = 100*(acc_by_class[il,0]/acc_by_class[il,1])
        print acc_by_class

    print '\n\nEpoch :  ' + str(epoch) + '\n\n'

    
    #save the model parameters
    torch.save(model.state_dict(), (save_path + save_name + '_' + str(epoch) +
                                    '_' + str(acc) + save_extension))


#FULL TEST
num_correct = 0
total = 0
for il, data in enumerate(testloader):
    #get the data
    batch_imgs,batch_labels = data
    if cuda:
        batch_imgs = batch_imgs.cuda()
    #forward pass 
    y_pred = model(Variable(batch_imgs))
    #class predictions equal argmax of output layer
    #TODO: use tensor.topk
    pred_class = torch.LongTensor(
                        [np.argmax(el) for el in 
                                    y_pred.cpu().data.numpy()])  

    #see how many were wrong (non-zero is wrong)
    diff = batch_labels - pred_class
    num_correct += len(diff) - np.count_nonzero(diff.numpy())
    total += len(diff) 
print 'FUll Test: Correct: {}  Total: {}  Accuracy: {}'.format(
                                        num_correct,
                                        total,
                                        num_correct/float(total))

