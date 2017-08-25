import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
from utils.blob import im_list_to_blob
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from vgg16_extractor import VGG16




class Part_Classifier(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [8, 16, 32]


    def __init__(self, classes=['background', 'object']):
        super(Part_Classifier, self).__init__()

        self.features = VGG16(bn=True)
        self._feat_stride = 16
        self.cross_entropy = None
        self.classes = classes
        self.num_classes = len(classes) 
        self.training = False 

        #self.classifier = FC(512,self.num_classes, relu=False)

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self, im_data,gt_boxes=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        features = self.features(im_data)

        #features = features.squeeze(0).permute(1,2,0)
        #features = features.contiguous().view(-1,features.size()[2])
        #class_scores = self.classifier(features)
        #true_class = gt_boxes[4]

        if self.training:
            self.build_loss(features, gt_boxes)

        return features

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x


        try:
            features.mean()
        except:
            print 'feat error 0'
    def build_loss(self,features, gt_boxes):

        #labels=torch.autograd.Variable(torch.LongTensor(int(gt_boxes[1][4])*
        #                               np.ones(features.size()[0]).astype(np.int64)))
        #labels = labels.cuda()
        #self.cross_entropy = F.cross_entropy(features,labels)

        #self.cross_entropy = network.np_to_variable(np.ones(1))

        #self.cross_entropy = features.mean()







#        gt_ind = np.random.choice(np.arange(gt_boxes.shape[0]-1)) + 1
#        gt_boxes = gt_boxes[[0,gt_ind], :]
#        gt_whs = np.array([gt_boxes[:,2]-gt_boxes[:,0], gt_boxes[:,3]-gt_boxes[:,1]]).transpose() /2
#        gt_centers = np.array([gt_boxes[:,0] + gt_whs[:,0] , gt_boxes[:,1]+gt_whs[:,1]]).transpose().astype(np.int32)
#
#        gt_centers = gt_centers/16
#
#        same_loss = torch.dist(features[0,:,gt_centers[1,1],gt_centers[1,0]],features[0,:,gt_centers[1,1]+1,gt_centers[1,0]])
#        diff_loss = torch.max(network.np_to_variable(np.zeros(1)) ,1 -torch.dist(features[0,:,gt_centers[1,1],gt_centers[1,0]],features[0,:,gt_centers[0,1],gt_centers[0,0]]))
#
#
#        self.cross_entropy = same_loss + diff_loss





        gt_boxes[:,2] -= 16
        gt_boxes[:,3] -= 16
        gt_ind = np.random.choice(np.arange(gt_boxes.shape[0]-1)) + 1
        gt_boxes = gt_boxes[[0,gt_ind], :]
        #gt_whs = np.array([gt_boxes[:,2]-gt_boxes[:,0], gt_boxes[:,3]-gt_boxes[:,1]]).transpose() /2
        #gt_centers = np.array([gt_boxes[:,0] + gt_whs[:,0] , gt_boxes[:,1]+gt_whs[:,1]]).transpose().astype(np.int32)

        #gt_centers = gt_centers/16

        #gt_one = features[0,:,gt_centers[1,1],gt_centers[1,0]]
        #gt_two = features[0,:,gt_centers[1,1],gt_centers[1,0]+1]
        #bg_one = features[0,:,gt_centers[0,1],gt_centers[0,0]]

        ##same_loss = torch.sum(torch.pow(gt_one,2)) - torch.sum(torch.pow(gt_two,2))
        ##diff_loss = torch.max(network.np_to_variable(np.zeros(1)) ,1-torch.sum(torch.pow(gt_one,2)) - torch.sum(torch.pow(bg_one,2)))

        #same_loss = F.smooth_l1_loss(gt_one, gt_two)
        #diff_loss = torch.max(network.np_to_variable(np.zeros(1)) ,1-F.smooth_l1_loss(gt_one, bg_one))

        #self.cross_entropy = 100*(same_loss + diff_loss)





        #convert boxes to locations in feature map: x,y,id
        gt_boxes[:,0:4] = gt_boxes[:,0:4].astype(np.int32)/self._feat_stride

        obj_x_locs = np.arange(gt_boxes[1,0],gt_boxes[1,2])
        bg_x_locs = np.arange(gt_boxes[0,0],gt_boxes[0,2])
        obj_y_locs = np.arange(gt_boxes[1,1],gt_boxes[1,3])
        bg_y_locs = np.arange(gt_boxes[0,1],gt_boxes[0,3])

        obj_x_locs,obj_y_locs = np.meshgrid(obj_x_locs,obj_y_locs)
        bg_x_locs,bg_y_locs = np.meshgrid(bg_x_locs,bg_y_locs)
        obj_locs = obj_y_locs.ravel()*(features.size()[3]-1) + obj_x_locs.ravel()
        bg_locs = bg_y_locs.ravel()*(features.size()[3]-1) + bg_x_locs.ravel()

        #print obj_locs.max()
        #print bg_locs.max()


        if obj_locs.shape[0] < 1:
            print 'Skip'
            return

        obj_loc_inds = np.random.choice(np.arange(obj_locs.shape[0]),2)
        bg_loc_ind = np.random.choice(np.arange(bg_locs.shape[0]))
        anchor_loc = obj_locs[obj_loc_inds[0]]
        pos_loc = obj_locs[obj_loc_inds[1]]
        bg_loc = bg_locs[bg_loc_ind] 

        features = features.squeeze(0).permute(1,2,0)
        features = features.contiguous().view(-1,features.size()[2])
        #feat_points = torch.index_select(features, 0 , torch.autograd.Variable(torch.LongTensor(gt_locations[:,0])).cuda())

        #print '{} {} {} {}'.format(features.size(), anchor_loc, pos_loc, bg_loc)

        anchor = features[anchor_loc,:].unsqueeze(0) 
        pos = features[pos_loc,:].unsqueeze(0) 
        neg = features[bg_loc,:].unsqueeze(0) 

        #print '{} {} {} {}'.format(features.size(), anchor.size(), pos.size(), neg.size())

        self.cross_entropy = F.triplet_margin_loss(anchor,pos,neg, margin=10, swap=True) * 10

        










        ##convert boxes to locations in feature map: x,y,id
        #gt_locations = []
        #for box in gt_boxes:
        #    #convert boxes to feature map coords
        #    #box = np.array([50, 50, 100, 100, 4])
        #    if box[2] > 1500 or box[3] > 1000:
        #        print 'skip {} {}'.format(box[2], box[3])
        #        continue
        #    box[0:4] = box[0:4]-1
        #    box[3] = min(self._feat_stride*features.size()[2] - 1, box[3])
        #    box[1] = min(box[1], box[3])
        #    box[2] = min(self._feat_stride*features.size()[3] - 1, box[2])
        #    box[0] = min(box[2], box[0])
        #    box[0:4] = box[0:4].astype(np.int32)/self._feat_stride

        #    x_locs = np.arange(box[0],box[2]+1)
        #    y_locs = np.arange(box[1],box[3]+1)
        #    x_locs,y_locs = np.meshgrid(x_locs,y_locs)
        #    #locs = np.vstack((x_locs.ravel(),y_locs.ravel())).transpose()
        #    locs = y_locs.ravel()*(features.size()[3]-1) + x_locs.ravel()
        #    #gt_locations.extend(np.hstack((locs,box[4]*np.ones(locs.shape[0]))))
        #    gt_locations.extend(np.vstack((locs,box[4]*np.ones(locs.shape[0]))).transpose())

        #gt_locations = np.asarray(gt_locations).astype(np.int64)
        #gt_locations = gt_locations[np.random.choice(np.arange(gt_locations.shape[0]), 10), :]
        #print gt_locations.max()
        #if gt_locations.max() > 8039:
        #    breakp =1
        #features = features.squeeze(0).permute(1,2,0)
        #features = features.contiguous().view(-1,features.size()[2])
        #feat_points = torch.index_select(features, 0 , torch.autograd.Variable(torch.LongTensor(gt_locations[:,0])).cuda())

        #for fp_ind in np.arange(feat_points.size()[0]-1):
        #    gt_class = gt_locations[fp_ind,1]
        #    if gt_class == 0:
        #        continue
        #    fp = feat_points[fp_ind,:]
        #    for fp_ind2 in np.random.choice(np.arange(gt_locations.shape[0]-1),10):
        #        gt_class2 = gt_locations[fp_ind2,1]
        #        fp2 = feat_points[fp_ind2,:]

        #        
        #        dist = torch.dist(fp,fp2)
        #        print dist
 
        #        if gt_class == gt_class2:
        #            self.cross_entropy += dist
        #            #print dist.data.cpu().numpy()
        #        else:
        #            self.cross_entropy += torch.max(network.np_to_variable(np.zeros(1)), 5 - dist)
        #            #print dist.data.cpu().numpy()

        #self.cross_entropy += torch.abs(feat_points.mean() / 100)

        #
        #print self.cross_entropy        

    def train(self):
        self.training = True
