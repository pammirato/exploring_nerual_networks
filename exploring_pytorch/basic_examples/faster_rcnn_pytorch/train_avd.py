import os
import torch
import torch.utils.data
import numpy as np
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD  
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
import exploring_pytorch.basic_examples.GetDataSet as GetDataSet


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
pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
#output_dir = 'models/saved_model3'
output_dir = ('/playpen/ammirato/Documents/exploring_neural_networks/exploring_pytorch' + 
             '/saved_models/')
save_name_base = 'FRA_1-5__2'


trained_model_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' +
                     'exploring_pytorch/saved_models/')
trained_model_name = 'faster_rcnn_voc07_trainval_100000.h5'
load_trained_model = False 
trained_epoch=0

start_step = 0
end_step = 100000
num_epochs = 20
lr_decay_steps = {60000, 80000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
use_tensorboard = False 
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
data_path = '/playpen/ammirato/Data/HalvedRohitData/'
#train_list=[
#             'Home_001_1',
#             'Home_001_2',
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
#CREATE TRAIN/TEST splits
train_set = GetDataSet.get_fasterRCNN_AVD(data_path,
                                          train_list,
                                          max_difficulty=5,
                                          chosen_ids=[0,1,2,3,4,5])

#create train/test loaders, with CUSTOM COLLATE function
trainloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=1,
                                          shuffle=True,
                                          collate_fn=AVD.collate)
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
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

for epoch in range(num_epochs):
    for step,batch in enumerate(trainloader):

        # get one batch
        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        gt_boxes = np.asarray(batch[1][0],dtype=np.float32)
        if gt_boxes.shape[0] == 0:#skip images with no gt boxes
            continue
        gt_boxes = gt_boxes[:,0:5]
        im_info = np.zeros((1,3))
        im_info[0,:] = [im_data.shape[1],im_data.shape[2],1]
        gt_ishard = np.zeros(gt_boxes.shape[0])
        dontcare_areas = np.zeros((0,4))

        # forward
        net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
        loss = net.loss + net.rpn.loss

        if _DEBUG:
            tp += float(net.tp)
            tf += float(net.tf)
            fg += net.fg_cnt
            bg += net.bg_cnt

        train_loss += loss.data[0]
        step_cnt += 1

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
            log_text = 'step %d, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
                step,  train_loss / step_cnt, fps, 1./fps)
            log_print(log_text, color='green', attrs=['bold'])

            if _DEBUG:
                log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
                log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                    net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
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
    if load_trained_model:
        save_name = os.path.join(output_dir, save_name_base+'_{}.h5'.format(
                                              epoch+trained_epoch))
    else:
        save_name = os.path.join(output_dir, save_name_base+'_{}.h5'.format(epoch))
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

