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
#from roi_pooling.modules.roi_pool import RoIPool
from vgg16_extractor import VGG16

import scipy.spatial.distance as sci_dist


class Patch_Detector(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [8, 16, 32]


    def __init__(self, classes=['background', 'object']):
        super(Patch_Detector, self).__init__()

        self.features = VGG16(bn=True)

        self._feat_stride = 16
        self.triplet_loss = None
        self.cross_entropy = None
        self.classes = classes
        self.num_classes =  2#len(classes) 
        self.training = False 

        #self.classifier = FC(512,self.num_classes, relu=False)

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self, im_data, anchor_data, target_box=None, other_boxes=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        img_features = self.features(im_data)

        anchor_data = network.np_to_variable(anchor_data, is_cuda=True)
        anchor_data = anchor_data.permute(0, 3, 1, 2)
        anchor_features = self.features(anchor_data)

        padding = (max(0,int(anchor_features.size()[2]/2)), 
                   max(0,int(anchor_features.size()[3]/2)))

        conved_feats = F.conv2d(img_features,anchor_features, padding=padding)
        

        if self.training:
            self.build_loss(conved_feats,target_box,other_boxes)

        return conved_feats 

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



    def build_loss(self, heat_map,target_box, other_boxes):

        heat_map = torch.div(heat_map, 512)


        if target_box.shape[0]>0:

            #convert target box to feature map coords
            target_id = target_box[4].astype(np.int64)
            target_box = (target_box[0:4] -1) / self._feat_stride
            target_box[0] = min(target_box[0], heat_map.size()[3]-1)
            target_box[1] = min(target_box[1], heat_map.size()[2]-1)
            target_box[2] = min(target_box[2], heat_map.size()[3]-1)
            target_box[3] = min(target_box[3], heat_map.size()[2]-1)
            target_box = target_box.astype(np.int32)
            target_box_center = [int((target_box[2]-target_box[0])/2.0+target_box[0]), 
                                 int((target_box[3]-target_box[1])/2.0+target_box[1])]
            
            #get target features from full image features
            target_heats = heat_map[0,:,target_box[1]:target_box[3]+1,
                                               target_box[0]:target_box[2]+1]


        #if there is another object in this image, get its features
        if other_boxes.shape[0]>0:

            #convert other boxes to feature map coords
            other_boxes = (other_boxes[0:4] -1) / self._feat_stride
            other_boxes[:,0] = np.fmin(other_boxes[:,0], heat_map.size()[3]-1)
            other_boxes[:,1] = np.fmin(other_boxes[:,1], heat_map.size()[2]-1)
            other_boxes[:,2] = np.fmin(other_boxes[:,2], heat_map.size()[3]-1)
            other_boxes[:,3] = np.fmin(other_boxes[:,3], heat_map.size()[2]-1)
            other_boxes = other_boxes.astype(np.int32)

            #get other features from full image
            #TODO:  get both bg and other objects
            o_ind = np.random.choice(np.arange(other_boxes.shape[0])) 
            other_heats = heat_map[0,
                                    :,
                                    other_boxes[o_ind,1]:other_boxes[o_ind,3]+1,
                                    other_boxes[o_ind,0]:other_boxes[o_ind,2]+1]
            other_box = other_boxes[o_ind,:]
            other_box_center = [int((other_box[2]-other_box[0])/2.0+other_box[0]), 
                                 int((other_box[3]-other_box[1])/2.0+other_box[1])]

        else:#otherwise get a hard mined negative bg features 
            
            #sort heat map vals
            np_heats = heat_map.data.cpu().numpy().squeeze()
            sort_by_row = np_heats.argsort()
            max_col_in_row = sort_by_row[:,-1] 


            possible_rows = np.arange(np_heats.shape[0])
            if target_box.shape[0]>0:
                #expand the box a bit
                close_to_target = [max(0,target_box[1]-1), 
                                   max(heat_map.size()[2],target_box[3]+2)]
                possible_rows = np.delete(possible_rows,np.arange(close_to_target[0],
                                                                  close_to_target[1]))
                
           
            #pick ten random inds 
            num_to_pick = np.fmin(possible_rows.size, 10)
            rows = np.random.choice(possible_rows,num_to_pick,replace=False)
            cols = max_col_in_row[rows]

            rowsv = Variable(torch.LongTensor(rows).cuda()) 
            colsv = Variable(torch.LongTensor(cols).cuda())
            #select those features from the total image features
            #first pick out all the rows and cols you want
            mhr = torch.index_select(heat_map,2,rowsv)
            mhrc = torch.index_select(mhr,3,colsv) 
            #then select the diagnol of the new feature map
            inds = np.arange(mhrc.size()[2])
            inds2 = np.expand_dims(np.expand_dims(np.matlib.repmat(inds,1,1),2),0)
            inds2v = Variable(torch.LongTensor(inds2.astype(np.int64)).cuda())
            other_heats = torch.gather(mhrc, 3, inds2v)


        target_loss = network.np_to_variable(np.zeros(1), is_cuda=True)
        #resize all features to be 512xN, N=num_patches
        if target_box.shape[0]>0:
            target_heats = target_heats.contiguous().view(1,-1)
            #target_loss = F.smooth_l1_loss(target_heats,target_label)
            ones = network.np_to_variable(np.ones(
                                     int(target_heats.size()[1])),is_cuda=True,
                                     dtype=torch.FloatTensor)
            target_label = network.np_to_variable(np.ones(
                                     int(target_heats.size()[1]))*-1,is_cuda=True,
                                     dtype=torch.FloatTensor)
            target_loss = torch.sum(torch.log(torch.add(ones,
                            torch.exp(torch.mul(target_heats,target_label)))))


        other_heats = other_heats.contiguous().view(1,-1)
        ones = network.np_to_variable(np.ones(
                                 int(other_heats.size()[1])),is_cuda=True,
                                 dtype=torch.FloatTensor)
        other_label = network.np_to_variable(np.ones(
                                 int(other_heats.size()[1]))*1,is_cuda=True, 
                                 dtype=torch.FloatTensor)
        other_loss = torch.sum(torch.log(torch.add(ones,
                        torch.exp(torch.mul(other_heats,other_label)))))
        self.cross_entropy  = target_loss + other_loss


        #target_mean = target_heats.mean()
        #other_mean = other_heats.mean()
        #margin = network.np_to_variable(np.array([100]), is_cuda=True)
        #zero = network.np_to_variable(np.zeros(1), is_cuda=True)      
 
        #self.cross_entropy = torch.max(zero, other_mean + margin - target_mean)


    def train(self):
        self.training = True

