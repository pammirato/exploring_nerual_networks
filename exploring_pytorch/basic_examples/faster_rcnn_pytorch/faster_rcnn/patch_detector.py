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

import scipy.spatial.distance as sci_dist


class Patch_Detector(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [8, 16, 32]


    def __init__(self, classes=['background', 'object']):
        super(Patch_Detector, self).__init__()

        self.features = VGG16(bn=True)
        self.img_patch_conv = Conv2d(512,512,5, same_padding=True)
        self.anchor_patch_conv = Conv2d(512,512,5)

        self._feat_stride = 16
        self.triplet_loss = None
        self.cross_entropy = None
        self.classes = classes
        self.num_classes =  2#len(classes) 
        self.training = False 

        self.classifier = FC(512,self.num_classes, relu=False)

    @property
    def loss(self):
        return self.cross_entropy * 10 + self.triplet_loss

    def forward(self, im_data, anchor_data, target_box=None, other_boxes=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        img_features = self.features(im_data)

        anchor_data = network.np_to_variable(anchor_data, is_cuda=True)
        anchor_data = anchor_data.permute(0, 3, 1, 2)
        anchor_features = self.features(anchor_data)

        img_patch_fts = self.img_patch_conv(img_features)
        anchor_patch_fts = self.anchor_patch_conv(anchor_features)

        #img_patch_fts = img_patch_fts.contiguous().view(512,-1).permute(1,0)
        img_patch_fts = img_patch_fts.contiguous().view(512,-1)
        anchor_patch_fts = anchor_patch_fts[0,:,0,:]      

        patch_fts = img_patch_fts + anchor_patch_fts.expand_as(img_patch_fts)
        patch_fts = patch_fts.permute(1,0)

 
        class_scores = self.classifier(patch_fts) 

        if self.training:
            self.build_loss(patch_fts,class_scores, img_features,anchor_features,target_box,other_boxes)

        return class_scores, img_features

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



    def build_loss(self, patch_features, class_scores, img_features, anchor_features, target_box, other_boxes):


        #convert target box to feature map coords
        target_id = target_box[4].astype(np.int64)
        target_box = (target_box[0:4] -1) / self._feat_stride
        target_box[0] = min(target_box[0], img_features.size()[3]-1)
        target_box[1] = min(target_box[1], img_features.size()[2]-1)
        target_box[2] = min(target_box[2], img_features.size()[3]-1)
        target_box[3] = min(target_box[3], img_features.size()[2]-1)
        target_box = target_box.astype(np.int32)
        target_box_center = [int((target_box[2]-target_box[0])/2.0+target_box[0]), 
                             int((target_box[3]-target_box[1])/2.0+target_box[1])]
        
        #get target features from full image features
        target_features = img_features[0,:,target_box[1]:target_box[3]+1,
                                           target_box[0]:target_box[2]+1]

        #print other_boxes.shape 

        other_random_indexes = True
        #if there is another object in this image, get its features
        if other_boxes.shape[0]>0:

            #convert other boxes to feature map coords
            other_classes = other_boxes[:,4]
            other_boxes = (other_boxes[0:4] -1) / self._feat_stride
            other_boxes[:,0] = np.fmin(other_boxes[:,0], img_features.size()[3]-1)
            other_boxes[:,1] = np.fmin(other_boxes[:,1], img_features.size()[2]-1)
            other_boxes[:,2] = np.fmin(other_boxes[:,2], img_features.size()[3]-1)
            other_boxes[:,3] = np.fmin(other_boxes[:,3], img_features.size()[2]-1)
            other_boxes = other_boxes.astype(np.int32)

            #get other features from full image
            #TODO:  get both bg and other objects
            o_ind = np.random.choice(np.arange(other_boxes.shape[0])) 
            other_id = other_classes[o_ind].astype(np.int32)
            other_features_X = img_features[0,
                                            :,
                                            other_boxes[o_ind,1]:other_boxes[o_ind,3]+1,
                                            other_boxes[o_ind,0]:other_boxes[o_ind,2]+1]
            other_box = other_boxes[o_ind,:]
            other_box_center = [int((other_box[2]-other_box[0])/2.0+other_box[0]), 
                                 int((other_box[3]-other_box[1])/2.0+other_box[1])]

        else:#otherwise get a hard mined negative bg features 
            #1. get all image features, target features
            #2. resize to NxD, D = feature vector length
            #3. get distances between all pairs of features            
            #4. get F feature vecs with min dist >0. F = #of target features

            #1
            np_img_features = img_features.data.cpu().numpy().squeeze()
            np_target_features = target_features.data.cpu().numpy()
           
            #2 
            np_img_features = np.reshape(np_img_features,(512,-1))
            np_img_features = np.swapaxes(np_img_features,1,0)
            np_target_features = np.reshape(np_target_features,(512,-1))
            np_target_features = np.swapaxes(np_target_features,1,0)
        
            #3 
            distances = sci_dist.cdist(np_target_features,np_img_features) 

            #4
            #get 2nd smallest distances for each target feature
            mins = np.partition(distances, 1)[:,1] 
            #get 2d indices of the image features that had the min distances 
            locs = [np.where(distances[il,:] == mins[il])[0][0] for il in range(len(mins))]
            locs2 = np.unravel_index(locs,(33,60))   
            rowsv = Variable(torch.LongTensor(locs2[0]).cuda()) 
            colsv = Variable(torch.LongTensor(locs2[1]).cuda())
            #select those features from the total image features
            #first pick out all the rows and cols you want
            mfeatr = torch.index_select(img_features,2,rowsv)
            mfeatrc = torch.index_select(mfeatr,3,colsv) 
            #then select the diagnol of the new feature map
            inds = np.arange(mfeatrc.size()[2])
            inds2 = np.expand_dims(np.expand_dims(np.matlib.repmat(inds,512,1),2),0)
            inds2v = Variable(torch.LongTensor(inds2.astype(np.int64)).cuda())
            other_features_X = torch.gather(mfeatrc, 3, inds2v)

            #make sure the hard mined pairs stay together
            other_random_indexes = False 
            other_id = 0
    
            other_box_center = [locs2[0][0], locs2[1][0]]

        #resize all features to be 512xN, N=num_patches
        anchor_features.squeeze()
        anchor_features = anchor_features.contiguous().view(512,-1)
        target_features = target_features.contiguous().view(512,-1)
        other_features_X = other_features_X.contiguous().view(512,-1)


        #choose training patch triplets (anchor, target, other)
        #Algorithm: (random)
        #Use as each feature vector once, but use as many as possible 
        #
        #1. Select how many patches to use(min of # of patches in anchor/target/other)
        #2. Generate random indexes for each(anchor/target/other)
        #3. select the random patches


        #first pick how many patches to use
        num_patches = min(target_features.size()[1], 
                           anchor_features.size()[1],
                           other_features_X.size()[1])

        #genereate random indexes
        anchor_indexes = np.random.permutation(
                          np.random.choice(
                            np.arange(anchor_features.size()[1]), 
                                        num_patches, replace=False))

        target_indexes = np.random.permutation(
                          np.random.choice(
                            np.arange(target_features.size()[1]), 
                                        num_patches, replace=False))

        if other_random_indexes:
            #choose random indexes for other too
            other_indexes = np.random.permutation(
                              np.random.choice(
                                np.arange(other_features_X.size()[1]),
                                            num_patches, replace=False))

        else:
            #choose same indexes for target and other
            other_indexes = target_indexes 

            



        #select the patches
        anchor_patches = torch.index_select(anchor_features,1,
                                Variable(torch.LongTensor(anchor_indexes).cuda()))
        target_patches = torch.index_select(target_features,1,
                                Variable(torch.LongTensor(target_indexes).cuda()))
        other_patches = torch.index_select(other_features_X,1,
                                Variable(torch.LongTensor(other_indexes).cuda()))


        anchor_patches = anchor_patches.permute(1,0)
        target_patches = target_patches.permute(1,0)
        other_patches = other_patches.permute(1,0)


        #print 'A: {}  T: {}  O: {}'.format(anchor_patches.mean().data.cpu().numpy(),
        #                                   target_patches.mean().data.cpu().numpy(),
        #                                   other_patches.mean().data.cpu().numpy())

        self.triplet_loss = F.triplet_margin_loss(anchor_patches, 
                                                  target_patches,
                                                  other_patches,
                                                  margin=3)




#        anchor_scores = self.classifier(anchor_patches)
#        target_scores = self.classifier(target_patches)
#        other_scores = self.classifier(other_patches)
#
#        #make gt labels
#        target_gt_labels = Variable(torch.LongTensor(np.ones(
#                                anchor_scores.size()[0]).astype(np.int64)*target_id).cuda())
#
#        #other_class = 0
#        #if other_boxes.shape[0] > 0:
#        #    other_class = other_boxes[o_ind,4]
#        other_gt_labels = Variable(torch.LongTensor(np.ones(
#                                anchor_scores.size()[0]).astype(np.int64)*other_id).cuda())
#
#
#        target_ce = F.cross_entropy(target_scores,target_gt_labels, size_average=False)
#        anchor_ce = F.cross_entropy(anchor_scores,target_gt_labels, size_average=False)
#        other_ce = F.cross_entropy(other_scores,other_gt_labels, size_average=False)
#
#        #print 'T: {} ACE: {} OCE: {}'.format(target_ce.data.cpu().numpy(),
#        #                                     anchor_ce.data.cpu().numpy(),
#        #                                     other_ce.data.cpu().numpy())
#        #print 'T: {} Other: {}'.format(target_id, other_id)
#
#        self.cross_entropy = target_ce + anchor_ce + other_ce


        target_label = network.np_to_variable(np.ones(1),is_cuda=True, dtype=torch.LongTensor)
        other_label = network.np_to_variable(np.zeros(1),is_cuda=True, dtype=torch.LongTensor)

        target_loc = target_box_center[1]*33 + target_box_center[0]
        other_loc = other_box_center[1]*33 + other_box_center[0]
        target_loc = network.np_to_variable(np.array([target_loc]),is_cuda=True,dtype=torch.LongTensor)
        other_loc = network.np_to_variable(np.array([other_loc]),is_cuda=True,dtype=torch.LongTensor)
        
        target_score = torch.index_select(class_scores,0,target_loc)
        other_score = torch.index_select(class_scores,0,other_loc)


        target_ce = F.cross_entropy(target_score,target_label)
        other_ce = F.cross_entropy(other_score,other_label)

        self.cross_entropy  = target_ce + other_ce


    def train(self):
        self.training = True

