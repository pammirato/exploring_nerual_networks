import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
from PreDefinedSquareImageNet_model10 import PreDefinedSquareImageNet
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#USER OPTIONS

#root directory of all scenes
data_path = '/playpen/ammirato/Data/RohitData/'
#where to save the model ater training
load_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' + 
             'exploring_pytorch/saved_models/recorded_models/' + 'model_17_2_0.880115469519.p')



chosen_ids = [0,4] #range(28) # [4,5,21,29]
combine_ids = [4,5,7,26]
num_classes = len(chosen_ids) #4  
#standard CNN inputs
learning_rate = .001

#standard CNN inputs
batch_size = 1 
#desired image size. HxWxC.Must be square => H=W
image_size = [32,32,3]

load_model = True



#CREATE TRAIN/TEST splits
##initialize transforms for the labels
# - add background bounding boxes to labels
rand_trans = AVD_transforms.AddRandomBoxes(
                            num_to_add=2,
                            box_dimensions_range=[100,100,200,200])
#only consider boxes from the chosen classes
pick_trans = AVD_transforms.PickInstances([-1])


#Make sure the boxes are valid(area>0, inside image)
validate_trans =AVD_transforms.ValidateMinMaxBoxes(min_box_dimensions=[10,10])

#convert the labels to tensors
to_tensor_trans = AVD_transforms.ToTensor()

#compose the transforms in a specific order, first to last
target_trans = AVD_transforms.Compose([#perturb_trans,
                                       rand_trans,  
                                       pick_trans,
                                       validate_trans,
                                       #ids_trans,
                                       to_tensor_trans])

##image transforms
#normalize image to be [-1,1]
norm_trans = AVD_transforms.NormalizePlusMinusOne()
#resize the images to be image_size
resize_trans = AVD_transforms.ResizeImage(image_size[0:2],'fill')
#compose the image transforms in a specific order
#TODO - make to_tensor last(i.e. make norm more general)
image_trans = AVD_transforms.Compose([resize_trans,
                                      to_tensor_trans,
                                      norm_trans])


#create the training/testing set objects
test_set = AVD.AVD_ByBox(root=data_path,
                         scene_list=['Home_001_1',
                                    'Home_001_2',
                                    'Home_008_1'],
                         transform=image_trans,
                         target_transform=target_trans,
                         classification=True)






##create train/test loaders, with CUSTOM COLLATE function
#trainloader = torch.utils.data.DataLoader(train_set,
#                                          batch_size=batch_size,
#                                          shuffle=True,
#                                          num_workers=2, 
#                                          collate_fn=AVD.collate)
testloader = torch.utils.data.DataLoader(test_set,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4, 
                                         collate_fn=AVD.collate)



#Define model, optimizer, and loss function
model = PreDefinedSquareImageNet(image_size,num_classes)
if load_model:
    model.load_state_dict(torch.load(load_path))




conv_filters = []
conv_biases = []
is_bias = 0
#organize model weights
for il,weights in enumerate(model.parameters()):

    weights = weights.data

    if is_bias:
        conv_biases.append(weights.numpy())
        is_bias = 0 

    if len(weights.size()) == 4: #conv layer
        conv_filters.append(weights.numpy())
        #next layer is the biases
        is_bias = 1


#weights_fig = plt.figure(1)
##set up outer grid, one per conv layer
#num_rows = int(np.ceil(np.sqrt(len(conv_filters))))
#num_cols = num_rows
#outer_grid = gridspec.GridSpec(num_rows,num_cols)
#outer_grid_index = 0 
##visualize weights
#for il in range(len(conv_filters)):
#    filters = conv_filters[il]
#    biases = conv_biases[il]
#
#    num_rows = int(np.ceil(np.sqrt(len(filters))))
#    num_cols = num_rows
#    inner_grid = gridspec.GridSpecFromSubplotSpec(
#                                num_rows,
#                                num_cols,
#                                subplot_spec=outer_grid[outer_grid_index]
#                                )
#    outer_grid_index += 1#for next layer
#
#    #for colormap
#    vmin = filters.min()
#    vmax = filters.max()
#    scale = 255.0/vmax
#
#
#    filters = np.transpose(filters,(2,3,1,0))
#    for jl in range(filters.shape[3]):
#        img = filters[:,:,:,jl]
#        bias = biases[jl]
#
#        img = (img + vmin) *scale
#
#        ax = plt.Subplot(weights_fig,inner_grid[jl]) 
#        ax.imshow(img.astype(np.uint8), vmin=vmin, vmax=vmax)
#        ax.set_title(str(bias))
#        weights_fig.add_subplot(ax)
#
#    ax = plt.Subplot(weights_fig,inner_grid[jl+1])
#    txt = ax.text(.1,.5,'Min: {}\n Max: {}'.format(vmin, vmax))
#    txt.set_clip_on(False)
#    weights_fig.add_subplot(ax)
#     
#
#
#plt.show()
#exit()

denorm_trans = AVD_transforms.DenormalizePlusMinusOne()
to_numpy_trans = AVD_transforms.ToNumpyRGB()

num_correct = 0
num_total = 0

acc_by_class = np.zeros((num_classes,3))


#VIS LOOP
for il,data in enumerate(testloader):
   
    #get the images and labels for this batch 
    batch_imgs,batch_labels = data
    
    #zero gradients, forward pass, compute loss
    pred, output_vis = model.forward_vis(Variable(batch_imgs))
    pred_score, pred_class = torch.nn.functional.softmax(pred).topk(1)
    if batch_labels[0] == pred_class.data.numpy()[0,0]:
        num_correct += 1
        acc_by_class[batch_labels[0],0] +=1
    num_total += 1

    acc_by_class[batch_labels[0],1] +=1

    fig = plt.figure(1)
  
    num_rows =1 
    num_cols =1 
 
    ax1 = fig.add_subplot(num_rows,num_cols,1) 
    
    denorm_img = denorm_trans(batch_imgs[0,:,:,:].cpu())
    org_img = to_numpy_trans(denorm_img) 
    ax1.imshow(org_img.astype(np.uint8))
    ax1.set_title('True: {} Pred: {} Score: {}'.format(batch_labels[0],
                                                       pred_class.data.numpy()[0][0],
                                                       pred_score.data.numpy()[0][0]))


    #maps = output_vis[1].data.cpu().squeeze(0).numpy()
    #maps = np.transpose(maps,(1,2,0))

    #for sub_index in range(maps.shape[2]):
    #    ax = fig.add_subplot(num_rows,num_cols,sub_index+2)
    #    ax.imshow(maps[:,:,sub_index], vmin=maps.min(), vmax = maps.max())
    #    sub_index +=1



    #cb = np.linspace(maps.min(),maps.max(), num=1000) 
    ##cb = np.reshape(np.asarray([ el/100.0 for el in range(700)]),(700,1))
    #cbw = np.tile(np.reshape(cb,(1000,1)),(1,1000))
    #ax = fig.add_subplot(num_rows,num_cols,sub_index+2); 
    #ax.imshow(cbw,vmin=maps.min(),vmax=maps.max());

    plt.draw()
    plt.pause(.001)

    raw_input('Press Enter: ')


for il in range(acc_by_class.shape[0]):
    acc_by_class[il,2] = 100*(acc_by_class[il,0]/acc_by_class[il,1])


print 'Correct: {}  Total {}  Percent: {}'.format(num_correct, num_total, 
                                                 num_correct/float(num_total)) 

print acc_by_class
