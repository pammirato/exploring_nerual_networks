import os
import torch
import torch.utils.data
import numpy as np
from datetime import datetime
import cv2

import matplotlib.pyplot as plt

from faster_rcnn import network
from faster_rcnn.faster_rcnn_target_driven import FasterRCNN, RPN
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
save_name_base = 'FRA_TD_1-5_archE_2'
save_freq = 1 

trained_model_path = ('/playpen/ammirato/Data/Detections/' +
                     '/saved_models/')
trained_model_name = 'FRA_TD_1-5_archE_2_6.h5'
load_trained_model = True 
trained_epoch  = 0

start_step = 0
end_step = 100000
num_epochs = 150
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
lr = cfg.TRAIN.LEARNING_RATE 
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
train_set = GetDataSet.get_fasterRCNN_AVD(data_path,
                                          train_list,
                                          #test_list,
                                          max_difficulty=4,
                                          chosen_ids=chosen_ids,
                                          by_box=False,
                                          fraction_of_no_box=.1)

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
target_path = '/playpen/ammirato/Data/big_bird_crops_80'
image_names = os.listdir(target_path)
image_names.sort()
#target_images = []
target_images ={} 
means = np.array([[[102.9801, 115.9465, 122.7717]]])
for name in image_names:
    target_data = cv2.imread(os.path.join(target_path,name))
    target_data = target_data - means
    target_data = np.expand_dims(target_data,axis=0)
    target_images[name[:-7]] = target_data


#imdb = get_imdb(imdb_name)
#rdl_roidb.prepare_roidb(imdb)
#roidb = imdb.roidb
#data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
#net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
net = FasterRCNN(classes=train_set.get_class_names(), debug=_DEBUG)

if load_trained_model:
    network.load_net(trained_model_path + trained_model_name, net)
else:
    network.weights_normal_init(net, dev=0.01)
    network.load_pretrained_npy(net, pretrained_model)
# model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
# model_file = 'models/saved_model3/faster_rcnn_60000.h5'
# network.load_net(model_file, net)
# exp_name = 'vgg16_02-19_13-24'
# start_step = 60001
# lr /= 10.
# network.weights_normal_init([net.bbox_fc, net.score_fc, net.fc6, net.fc7], dev=0.01)



net.cuda()
net.train()

params = list(net.parameters())
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
#optimizer = torch.optim.Adam(params[8:], lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
tp, tf, fg, bg = 0., 0., 1,1 
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()



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
        if gt_boxes.shape[0] == 0:
            gt_boxes = np.asarray([[0,0,1,1,0]])
 
        #get the gt inds that are in this image, not counting 0(background)
        objects_present = gt_boxes[:,4]
        objects_present = objects_present[np.where(objects_present!=0)[0]]

        thresh = use_not_present_target
        #if objects_present.shape[0] > 0:
        #    counts = np.array([count_by_class[x] for x in objects_present])
        #    present_probs = counts / float(sum(all_class_counts))
        #    thresh = min(thresh, use_not_present_target_coef*min(present_probs))

        #pick a random target, with a bias towards choosing a target that 
        #is in the image. Also pick just one gt_box, since there is one target
        if np.random.rand() > thresh and objects_present.shape[0]!=0:

            #heuristic to balance classes, bias towards picking
            #classes with less examples
            #if objects_present.shape[0] > 1:
            #    counts = np.array([count_by_class[x] for x in objects_present])
            #    counts = counts - .6 * min(counts)
            #    probs = 1 -  (counts/float(sum(counts)))
            #    probs = probs / sum(probs)
            #else:
            #    probs = [1]
            #target_ind = int(np.random.choice(objects_present, p=probs))
            
            target_ind = int(np.random.choice(objects_present))
            gt_boxes = gt_boxes[np.where(gt_boxes[:,4]==target_ind)[0],:-1]
            gt_boxes[0,4] = 1

            tv_cnt += 1
            targets_cnt[target_ind][0] += 1 
        else:#the target is not in the image, give a dummy background box
            not_present = np.asarray([ind for ind in chosen_ids 
                                          if ind not in objects_present and 
                                             ind != 0]) 
            target_ind = int(np.random.choice(not_present))
            gt_boxes = np.asarray([[0,0,1,1,0]])

        #target_data = target_images[target_ind-1]
        target_data = target_images[id_to_name[target_ind]]
        targets_cnt[target_ind][1] += 1 


        im_info = np.zeros((1,3))
        im_info[0,:] = [im_data.shape[1],im_data.shape[2],1]
        gt_ishard = np.zeros(gt_boxes.shape[0])
        dontcare_areas = np.zeros((0,4))

        # forward
        ir_cnt +=1
        net(target_data,im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
        loss = net.loss + net.rpn.loss
        loss *= class_weights[target_ind]
        #if _DEBUG:
            #tp += float(net.tp)
            #tf += float(net.tf)
            #fg += net.fg_cnt
            #bg += net.bg_cnt

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

            #log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            #    step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
            #log_text = 'step %d, loss: %.4f, fps: %.2f (%.2fs per batch) tv_cnt:%d' \
            #           'ir_cnt:%d epoch:%d' % (
            #    step,  train_loss / step_cnt, fps, 1./fps, tv_cnt, ir_cnt, epoch)
            log_text = 'step %d,epoc_avg_loss: %.4f, fps: %.2f (%.2fs per batch) tv_cnt:%d' \
                       'ir_cnt:%d epoch:%d loss: %.4f tot_avg_loss: %.4f' % (
                step,  epoch_loss/(step+1), fps, 1./fps, tv_cnt, ir_cnt, epoch, loss.data[0],
                train_loss/step_cnt)
            log_print(log_text, color='green', attrs=['bold'])
            if step%(disp_interval*10) == 0:
                for cid in targets_cnt.keys():
                    print '{}: {} / {}'.format(cid,targets_cnt[cid][0],targets_cnt[cid][1])

            if _DEBUG:
                log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
                log_print('\trpn_cls: %.4f, rpn_box: %.4f, rpn_feat: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                    net.rpn.cross_entropy.data.cpu().numpy()[0], 
                    net.rpn.loss_box.data.cpu().numpy()[0],
                    net.rpn.feature_extraction_loss.data.cpu().numpy()[0],
                    net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
                )
            re_cnt = True

        if use_tensorboard and step % log_interval == 0:
            exp.add_scalar_value('train_loss', train_loss / step_cnt, step=step)
            exp.add_scalar_value('learning_rate', lr, step=step)
            if _DEBUG:
                exp.add_scalar_value('true_positive', tp/fg*100., step=step)
                exp.add_scalar_value('true_negative', tf/bg*100., step=step)
                losses = {'rpn_cls': float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                          'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
                          'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
                          'rcnn_box': float(net.loss_box.data.cpu().numpy()[0])}
                exp.add_scalar_dict(losses, step=step)

    #epoch over
    if epoch % save_freq == 0:
        if load_trained_model:
            save_name = os.path.join(output_dir, save_name_base+'_{}.h5'.format(
                                                  epoch+trained_epoch+1))
        else:
            save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}.h5'.format(epoch, epoch_loss/(step+1)))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))


#    if step in lr_decay_steps:
#        lr *= lr_decay
#        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
#
#    if re_cnt:
#        tp, tf, fg, bg = 0., 0., 0, 0
#        train_loss = 0
#        step_cnt = 0
#        t.tic()
#        re_cnt = False

