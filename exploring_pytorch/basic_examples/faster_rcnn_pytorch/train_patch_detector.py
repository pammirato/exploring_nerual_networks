import os
import torch
import torch.utils.data
import numpy as np
from datetime import datetime
import cv2

#import matplotlib.pyplot as plt

from faster_rcnn import network
#from faster_rcnn.faster_rcnn_target_driven import FasterRCNN, RPN
from faster_rcnn.patch_detector import Patch_Detector
#from faster_rcnn.faster_rcnn_target_driven import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD  
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
import exploring_pytorch.basic_examples.GetDataSet as GetDataSet


#TODO make target image to gt_box index(id) more robust,clean, better


try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
#pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
pretrained_model = '/playpen/ammirato/Data/Detections/pretrained_models/VGG_imagenet.npy'
#output_dir = 'models/saved_model3'
output_dir = ('/playpen/ammirato/Data/Detections/' + 
             '/saved_models/')
save_name_base = 'PD_1-5_archA_6'
save_freq = 1 

trained_model_path = ('/playpen/ammirato/Data/Detections/' +
                     '/saved_models/')
trained_model_name = 'PD_1-5_archA_0_9_69.38499.h5'
load_trained_model = False 
trained_epoch = 9 

start_step = 0
end_step = 100000
num_epochs = 50
lr_decay_steps = {60000, 80000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
use_tensorboard = False 
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard


use_not_present_target = .4

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE *.25
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval =10# cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
data_path = '/playpen/ammirato/Data/HalvedRohitData/'
#train_list=[
#             'Home_001_1', #             'Home_001_2',
#            'Home_003_1',
#             'Home_003_2',
#             'Home_004_1',
#             'Home_004_2',
#             'Home_005_1',
#             'Home_005_2',
#            'Home_014_1',
#             'Home_014_2',
#            ]
train_list=[
             'Home_001_1',
             'Home_001_2',
             'Home_002_1',
             'Home_004_1',
             'Home_004_2',
             'Home_005_1',
             'Home_005_2',
             'Home_006_1',
             'Home_008_1',
             'Home_014_1',
             'Home_014_2',
            ]


test_list=[
            'Home_003_1',
          ]

chosen_ids = range(6)
#CREATE TRAIN/TEST splits
train_set = GetDataSet.get_part_classifier_AVD(data_path,
                                          train_list,
                                          #test_list,
                                          max_difficulty=4,
                                          chosen_ids=chosen_ids,
                                          by_box=False,
                                          fraction_of_no_box=0,
                                          add_background=False)

#create train/test loaders, with CUSTOM COLLATE function
trainloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=1,
                                          shuffle=True,
                                          collate_fn=AVD.collate)



id_to_name = GetDataSet.get_class_id_to_name_dict(data_path)
count_by_class = train_set.get_count_by_class()
all_class_counts = np.array([count_by_class[x] for x in count_by_class.keys()])
class_probs = all_class_counts / float(sum(all_class_counts))
use_not_present_target_coef = len(count_by_class.keys()) * use_not_present_target

#compute class weights
total = float(sum(all_class_counts))
class_weights = {} 
max_weight = 0 
for cid in count_by_class.keys():
    if count_by_class[cid] > 0:
        #TODO: make more robust. use class id, not index
        wt =  total / count_by_class[cid]
        class_weights[cid] = wt 
        if wt > max_weight:
            max_weight = wt

for cid in class_weights.keys():
    class_weights[cid] = class_weights[cid]/max_weight



#load all target images
target_path = '/playpen/ammirato/Data/big_bird_patches_80'
image_names = os.listdir(target_path)
image_names.sort()
target_images ={} 
means = np.array([[[102.9801, 115.9465, 122.7717]]])
for name in image_names:
    target_data = cv2.imread(os.path.join(target_path,name))
    target_data = target_data - means
    target_data = np.expand_dims(target_data,axis=0)
    target_images[name[:-11]] = target_data



# load net
net = Patch_Detector(classes=train_set.get_class_names())

if load_trained_model:
    network.load_net(trained_model_path + trained_model_name, net)
else:
    network.weights_normal_init(net, dev=0.01)
    network.load_pretrained_pc(net, pretrained_model)



net.cuda()
net.train()

params = list(net.parameters())
#optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(params, lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
tp, tf, fg, bg = 0., 0., 1,1 
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

cur_el = 0
prev_el = 0
prev2_el = 0


for epoch in range(num_epochs):
    tv_cnt = 0
    ir_cnt = 0
    targets_cnt = {} 
    epoch_loss = 0
    for cid in id_to_name.keys():
        targets_cnt[cid] = [0,0]
    for step,batch in enumerate(trainloader):

        # get one batch
        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        gt_boxes = np.asarray(batch[1][0],dtype=np.float32)
 


        if gt_boxes.shape[0] < 1:
            print 'SKIPPPPPPPPPPPPP'
            continue

        #if gt_boxes.shape[0] > 1:
            #print 'BIGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG'
            #print gt_boxes
            #continue

        #get the gt inds that are in this image, not counting 0(background)
        objects_present = gt_boxes[:,4]
        objects_present = objects_present[np.where(objects_present!=0)[0]]






        thresh = use_not_present_target

        target_box = np.zeros(0) 
        other_boxes = np.zeros(0) 

        #pick a random target, with a bias towards choosing a target that 
        #is in the image. Also pick just one gt_box, since there is one target
        if np.random.rand() > thresh and objects_present.shape[0]!=0:

            target_ind = int(np.random.choice(objects_present))

            target_box_ind = np.where(gt_boxes[:,4]==target_ind)[0]
            other_inds = np.where(gt_boxes[:,4] != target_ind)[0]
            target_box =  gt_boxes[target_box_ind,:-1].squeeze()
            other_boxes =  np.delete(gt_boxes,target_box_ind,axis=0)

            #gt_boxes = gt_boxes[np.where(gt_boxes[:,4]==target_ind)[0],:-1]
            #gt_boxes[0,4] = 1 

            tv_cnt += 1
            targets_cnt[target_ind][0] += 1 
        else:#the target is not in the image, give a dummy background box
            not_present = np.asarray([ind for ind in chosen_ids 
                                          if ind not in objects_present and 
                                             ind != 0]) 
            target_ind = int(np.random.choice(not_present))
            other_boxes = gt_boxes
            #gt_boxes = np.asarray([[0,0,1,1,0]])

        #target_data = target_images[target_ind-1]
        target_data = target_images[id_to_name[target_ind]]
        targets_cnt[target_ind][1] += 1 











        #choose a target object
        #target_ind = int(np.random.choice(objects_present))
        #target_box_ind = np.where(gt_boxes[:,4]==target_ind)[0]
        #other_inds = np.where(gt_boxes[:,4] != target_ind)[0]
        ##target_box =  gt_boxes[target_box_ind,:-1]
        #target_box =  gt_boxes[target_box_ind,:-1].squeeze()
        #other_boxes =  np.delete(gt_boxes,target_box_ind,axis=0)

        #get target anchor image
        target_data = target_images[id_to_name[target_ind]]



        # forward
        net(im_data,
            anchor_data=target_data, 
            target_box=target_box, 
            other_boxes=other_boxes)

        loss = net.loss 
        #loss *= class_weights[gt_boxes[4]]

        train_loss += loss.data[0]
        step_cnt += 1
        epoch_loss += loss.data[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        network.clip_gradient(net, 10.)
        optimizer.step()



        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = step_cnt / duration
            log_text = 'step %d,epoc_avg_loss: %.4f, fps: %.2f (%.2fs per batch) ' \
                       'epoch:%d loss: %.4f tot_avg_loss: %.4f ce_loss: %.4f t_loss: %.4f' % (
                step,  epoch_loss/(step+1), fps, 1./fps,  epoch, loss.data[0],
                train_loss/step_cnt, net.cross_entropy.data[0], net.cross_entropy.data[0])
            log_print(log_text, color='green', attrs=['bold'])
            #if step%(disp_interval*10) == 0:
            #    for cid in targets_cnt.keys():
            #        print '{}: {} / {}'.format(cid,targets_cnt[cid][0],targets_cnt[cid][1])


    #epoch over
    if epoch % save_freq == 0:
        if load_trained_model:
            save_name = os.path.join(output_dir, save_name_base+'_{}.h5'.format(
                                                  epoch+trained_epoch+1))
        else:
            save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}.h5'.format(epoch, epoch_loss/(step+1)))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))

    prev2_el = prev_el
    prev_el = cur_el
    cur_el = epoch_loss/(step+1)

    if prev2_el >0 and cur_el-prev_el >-.1 and prev_el-prev2_el >-.1:
        lr = .5*lr
        #optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(params, lr=lr)
