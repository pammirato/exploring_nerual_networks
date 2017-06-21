import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
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
save_name = 'model_38'
load_name = 'model_38_18_0.781119076436.p'
save_extension = '.p'
#which/how many classes to learn
chosen_ids = range(28) #[0,2,5,6,7,10,12,13,14,19,22,27]
combine_ids = [4,5,7,26]
max_difficulty = 4 
num_classes = len(chosen_ids) #4 
#standard CNN inputs
learning_rate = .00005
batch_size = 128 
max_epochs = 50  
use_class_weights = True 
use_added_loss = True 

#desired image size. HxWxC.Must be square => H=W
image_size = [224,224,3]
org_img_dims = [1920/2, 1080/2]
#whether to use the GPU
cuda = True 

preload_images = True 
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







##initialize transforms for the labels
# - add background bounding boxes to labels
back_trans = AVD_transforms.AddBackgroundBoxes(
                            num_to_add=2,
                            box_dimensions_range=[50,50,300,300],
                            image_dimensions=org_img_dims)
#only consider boxes from the chosen classes
pick_trans = AVD_transforms.PickInstances(chosen_ids,
                                          max_difficulty=max_difficulty)
#add more boxes in each image, each randomly perturbed from original
perturb_trans = AVD_transforms.AddPerturbedBoxes(num_to_add=3,
                                                 changes = [[-.30,.40],
                                                           [-.30,.40], 
                                                           [-.40,.30], 
                                                           [-.40,.30]],
                                                percents=True)
#                                                changes = [[-50,20],
#                                                           [-50,20], 
#                                                           [-20,50], 
#                                                           [-20,50]])


#Make sure the boxes are valid(area>0, inside image)
validate_trans =AVD_transforms.ValidateMinMaxBoxes(min_box_dimensions=[10,10],
                                                image_dimensions=org_img_dims)
#make class ids consecutive (see transforms docs)
ids_trans = AVD_transforms.MakeIdsConsecutive(chosen_ids)
#convert the labels to tensors
to_tensor_trans = AVD_transforms.ToTensor()

#combine boxes to one class
combine_trans = AVD_transforms.CombineInstances(combine_ids)


#compose the transforms in a specific order, first to last
target_trans = AVD_transforms.Compose([
                                       #combine_trans,
                                       pick_trans,
                                       perturb_trans,
                                       back_trans,  
                                       validate_trans,
                                       ids_trans,
                                       to_tensor_trans])

##image transforms
#normalize image to be [-1,1]
#norm_trans = AVD_transforms.NormalizePlusMinusOne()
norm_trans = AVD_transforms.NormalizeRange(0.0,1.0)
norm_trans2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#resize the images to be image_size
resize_trans = AVD_transforms.ResizeImage(image_size[0:2],'fill')
#compose the image transforms in a specific order
#TODO - make to_tensor last(i.e. make norm more general)
image_trans = AVD_transforms.Compose([resize_trans,
                                      to_tensor_trans,
                                      norm_trans,
                                      norm_trans2])


back_trans = AVD_transforms.AddBackgroundBoxes(
                            num_to_add=1,
                            box_dimensions_range=[100,100,200,200],
                            image_dimensions=org_img_dims)



test_target_trans = AVD_transforms.Compose([#perturb_trans,
                                       back_trans,  
                                       pick_trans,
                                       validate_trans,
                                       ids_trans,
                                       to_tensor_trans])





#CREATE TRAIN/TEST splits
if reload_train_test:



    #create the training/testing set objects
    #train_set = AVD.AVD_ByBox(root=data_path,
    #                          scene_list=['Home_014_1',
    #                                      'Home_014_2',
    #                                      'Home_003_1',
    #                                      'Home_003_2',
    #                                      'Home_002_1'],
    #                          transform=image_trans,
    #                          target_transform=target_trans,
    #                          classification=True)
    train_set = AVD.AVD_ByBox(root=data_path,
                             scene_list=[
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
                                        ],
                             train=True,
                             transform=image_trans,
                             target_transform=target_trans, 
                             classification=True,
                             preload_images=preload_images)


    #test_set = AVD.AVD_ByBox(root=data_path,
    #                         scene_list=['Home_001_1',  
    #                                    'Home_001_2',
    #                                    'Home_005_1'],
    #                         transform=image_trans,
    #                         target_transform=target_trans,
    #                         classification=True)
    test_set = AVD.AVD_ByBox(root=data_path,
                             train=False,
                              scene_list=[
                                          'Home_006_1',
                                          'Home_008_1',
                                          'Home_002_1'
                                         ],
                             transform=image_trans,
                             target_transform=test_target_trans,
                             classification=True,
                             preload_images=preload_images)



if reassign_transforms:
    train_set.transform = image_trans
    train_set.target_transform = target_trans
    test_set.transform = image_trans
    test_set.target_transform = test_target_trans
     


batch = train_set[0:3]


#create train/test loaders, with CUSTOM COLLATE function
trainloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=batch_size,
                                          shuffle=True,
#                                          num_workers=1, 
                                          collate_fn=AVD.collate)
testloader = torch.utils.data.DataLoader(test_set,
                                         batch_size=batch_size,
                                         shuffle=True,
#                                         num_workers=4, 
                                         collate_fn=AVD.collate)





#get class weights
count_by_class = train_set.get_count_by_class(num_classes)
total = len(train_set)

class_weights = np.zeros(num_classes)
#class_weights = []


for il in range(num_classes):
    if count_by_class[il] > 0:
        class_weights[il] =  total / count_by_class[il]
        #class_weights.append(torch.FloatTensor([total/count_by_class[il]]).cuda())
    #else:
    #    class_weights.append(torch.FloatTensor([0]))
        
class_weights = torch.FloatTensor(class_weights)
class_weights = class_weights.cuda()

#Define model, optimizer, and loss function
model = PreDefinedSquareImageNet(image_size,num_classes)
if load_model and not(use_pretrained_alexnet):
    model.load_state_dict(torch.load(load_path + load_name))
if use_pretrained_alexnet:
    model = models.alexnet(pretrained=True)
    #change ouput layer 
    model.classifier._modules['6'] = torch.nn.modules.linear.Linear(4096,num_classes) 

    if load_model:
        model.load_state_dict(torch.load(load_path + load_name))

    #freeze all layers
    #as in http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward
    counter = 0
    params_to_optimize = []
    for param in model.parameters():
        if counter > 9:
            params_to_optimize.append(param)
            continue
#        param.requires_grad = False
        counter +=1
   
    

if cuda:
    model.cuda()

if use_pretrained_alexnet:
    #optimizer = torch.optim.Adam(model.classifier._modules['6'].parameters(),lr=learning_rate)
    #optimizer = torch.optim.Adam(params_to_optimize,lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
else:
    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#loss_fn = torch.nn.NLLLoss()

if use_class_weights:
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
else:
    loss_fn = torch.nn.CrossEntropyLoss()

if use_added_loss:
    loss_fns = [torch.nn.CrossEntropyLoss(weight=class_weights),
                torch.nn.CrossEntropyLoss()]

#TRAIN LOOP
for epoch in range(max_epochs):
#    clock = Timer()
    for il,data in enumerate(trainloader):
      

#        clock.tic() 
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
       
#        print clock.toc(average=False) 


        #show loss every few iterations 
        if il % show_loss_iter == 0:
            print 'Iter: {}  Loss : {} '.format(il,loss.data[0])    
            print batch_imgs[0].cpu().min()

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
            pred_class = torch.LongTensor(
                                [np.argmax(el) for el in 
                                            y_pred.cpu().data.numpy()])  

            #see how many were wrong (non-zero is wrong)
            diff = batch_labels - pred_class
            num_correct += len(diff) - np.count_nonzero(diff.numpy())
            total += len(diff) 

            #keep track of num correcrt/total for each class
            for jl in range(batch_labels.size()[0]):
                gt = batch_labels[jl]
                pred = pred_class[jl]
                acc_by_class[gt,1] +=1
                if(gt == pred):
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

