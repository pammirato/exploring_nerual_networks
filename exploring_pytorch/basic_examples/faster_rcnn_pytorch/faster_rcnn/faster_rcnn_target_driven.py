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
from rpn_msr.rois_target_layer import rois_target_layer as rois_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from vgg16 import VGG16


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class RPN(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [8, 16, 32]

    def __init__(self):
        super(RPN, self).__init__()

        #first 5 conv layers of VGG? only resizing is 4 max pools
        self.features = VGG16(bn=False)

        self.target_conv = Conv2d(512,512,5)
        self.target_embedding = Conv2d(512+512, 512, 1);

        self.conv1 = Conv2d(512,512, 5, same_padding=True)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1, relu=False, same_padding=False)

        # loss
        self.cross_entropy = None
        self.loss_box = None
        self.feature_extraction_loss = None

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10 + self.feature_extraction_loss*10

    def forward(self, target_data, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
      
        #get image features 
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        features = self.features(im_data)

        #get target image features
        target_data = network.np_to_variable(target_data, is_cuda=True)
        target_data = target_data.permute(0, 3, 1, 2)
        target_features = self.features(target_data)

        #subtract target_features from each image feature location
        ts = target_features.size()[3]
        num_feats = target_features.size()[1]
        ones_kernel = network.np_to_variable(np.ones((1,num_feats,ts,ts)))
        summed_image_feats = F.conv2d(features,ones_kernel, padding=int(ts/2))
        summed_target_feats = F.conv2d(target_features,ones_kernel) 
        feat_diffs = summed_image_feats - summed_target_feats.expand_as(summed_image_feats)
        squared_diffs = feat_diffs.pow(2)
        rooted_diffs = squared_diffs.sqrt()
        div_diffs = rooted_diffs.div(ts*num_feats)
        #neg_diffs = div_diffs.sub(float(div_diffs.max().data.cpu().numpy()[0]))
        #reflected_diffs = neg_diffs.mul(-1)
        #normed_diffs = reflected_diffs 


        target_features_reduced = self.target_conv(target_features) 
        ##concat and embed target features with features
        target_features_expand = target_features_reduced.expand_as(features)
        x = torch.cat([features,target_features_expand],1)
        x = self.target_embedding(x)
       
 
        rpn_conv1 = self.conv1(x)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales)*3*2)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois,scores,anchor_inds=self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                                 cfg_key, self._feat_stride, self.anchor_scales,
                                      return_scores=True, return_anchor_inds=True)

        max_score = scores.max()

        # generating training labels and build the rpn loss
        if self.training:
            self.feature_extraction_loss = self.feature_extraction_loss_layer(
                                                div_diffs, features, target_features, 
                                                gt_boxes,
                                                self._feat_stride)
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_score,gt_boxes, gt_ishard, dontcare_areas,
                                                im_info, self._feat_stride, self.anchor_scales)
            #rpn_data = self.rois_target_layer(rois,gt_boxes, gt_ishard, dontcare_areas,
            #                                    im_info, self._feat_stride, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
            #self.cross_entropy, self.loss_box = self.build_loss2(rpn_cls_score, rpn_bbox_pred, rpn_data,anchor_inds, scores)

        return target_features, features, rois, scores



    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        #ORIGINAL
        # classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_label = rpn_data[0].view(-1)

        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        fg_cnt = torch.sum(rpn_label.data.ne(0))

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, size_average=False)
        #rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

        # box loss
        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights) 
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return rpn_cross_entropy, rpn_loss_box
        #return network.np_to_variable(np.array([0])), network.np_to_variable(np.array([0]))




    @staticmethod
    def feature_extraction_loss_layer(normed_diffs, image_features, 
                                      target_features, gt_box, feat_stride):
       
        target_feat_loss = network.np_to_variable(np.zeros(1))
        bg_loss = network.np_to_variable(np.zeros(1))

        gt_center = []
         
        #make sure gt_box is valid
        if gt_box is not None and gt_box.shape[0]>0 and gt_box[0,4] > 0:
            gt_box = gt_box[0,:]
            #get gt box center in feature map coordinates
            gt_feat_coords = gt_box/feat_stride
            gt_center = [.5*gt_feat_coords[2] + .5*gt_feat_coords[0],
                         .5*gt_feat_coords[3] + .5*gt_feat_coords[1]]

            gt_center[0] = min(gt_center[0], normed_diffs.size()[3]-1)
            gt_center[1] = min(gt_center[1], normed_diffs.size()[2]-1)
           
            #pick out a box of features, same size as target features
            gt_center = np.floor(gt_center)
            gt_center = [int(gt_center[0]), int(gt_center[1])]

            target_feat_loss = (normed_diffs[0,0,gt_center[1],gt_center[0]])

        #hard-mine negative examples
        feat_diffs = normed_diffs.squeeze().data.cpu().numpy()
        #get rid of boundaries #TODO
        feat_diffs = np.abs(feat_diffs[2:-2,2:-2])
        min_diffs = feat_diffs.min(0) 
        min_row_indexes = feat_diffs.argmin(0)  + 2
        min_col_indexes = np.arange(min_diffs.shape[0]) + 2
        #don't consider places with the gt box
        if gt_center and gt_center[0] < min_diffs.shape[0]:
            min_diffs = np.delete(min_diffs, gt_center[0])
            min_row_indexes = np.delete(min_row_indexes, gt_center[0]) 
            min_col_indexes = np.delete(min_col_indexes, gt_center[0]) 

        #chose globalish min
        chosen_min_row = min_row_indexes[min_diffs.argmin()]
        chosen_min_col = min_col_indexes[min_diffs.argmin()]


        bg_feat_loss = torch.max(network.np_to_variable(np.array([0])), 
                                 1 - normed_diffs[0,0,chosen_min_row,chosen_min_col])



        reg_loss = torch.max(network.np_to_variable(np.array([0])), 
                            1 - image_features.mean())
        reg_loss += torch.max(network.np_to_variable(np.array([0])), 
                            1 - target_features.mean())

        #return  target_feat_loss + bg_feat_loss + reg_loss
        return  target_feat_loss 





    def build_loss2(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data, anchor_inds, scores):
        # classification loss

        #have to reshape rpn_cls_score the same way we did the rpn_cls_prob
        #during proposal_layer to get the rois
        #rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()
        bg_scores = rpn_cls_score_reshape[:,:9,:,:]
        fg_scores = rpn_cls_score_reshape[:,9:,:,:]
        bg_scores = bg_scores.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        fg_scores = fg_scores.permute(0, 2, 3, 1).contiguous().view(-1, 1)

        rpn_cls_score  = torch.cat([bg_scores,fg_scores],1)

        #rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_label = rpn_data[0].view(-1)


        anchor_keep = Variable(torch.LongTensor(anchor_inds)).cuda()
        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, anchor_keep)

        cls_probs = F.softmax(rpn_cls_score)
        diff = cls_probs[:,1] - scores[:,1]
        #print 'DIFF {}'.format(diff.max().data.cpu().numpy())
        assert(diff.max() < .1)       
 
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        fg_cnt = torch.sum(rpn_label.data.ne(0))
        #print fg_cnt

        #weight = [torch.FloatTensor([0]), torch.FloatTensor([10])]
        #weight = torch.FloatTensor([0,10])
        #weight = weight.cuda()
        #rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, weight=weight)
        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, size_average=False)
        #rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

        # box loss
        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = rpn_bbox_pred.view(-1,4)
        rpn_bbox_pred = torch.index_select(rpn_bbox_pred,0,anchor_keep)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return rpn_cross_entropy, rpn_loss_box

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

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales, return_scores=False, return_anchor_inds=False):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        if return_scores:
            x,y,inds = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales,return_scores=return_scores, return_anchor_inds=return_anchor_inds)
            x = network.np_to_variable(x, is_cuda=True)
            #add 0's for bg class score
            z = np.zeros((x.size()[0], 2))
            z[:,1] = y[:,0]
            z = network.np_to_variable(z,is_cuda=True)
            return x.view(-1, 5), z, inds
        else:
            x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales,return_scores=return_scores)
            x = network.np_to_variable(x, is_cuda=True)
            return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales)

        rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
        rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
        rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    @staticmethod
    def rois_target_layer(rois, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            rois_target_layer_py(rois, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales)

        rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
        rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
        rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        self.features.load_from_npz(params)

        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv': 'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)


class FasterRCNN(nn.Module):
    ###n_classes = 21
    ###classes = np.asarray(['__background__',
    ###                   'aeroplane', 'bicycle', 'bird', 'boat',
    ###                   'bottle', 'bus', 'car', 'cat', 'chair',
    ###                   'cow', 'diningtable', 'dog', 'horse',
    ###                   'motorbike', 'person', 'pottedplant',
    ###                   'sheep', 'sofa', 'train', 'tvmonitor'])
    n_classes = 2
    classes = np.asarray(['__background__', 'object'])

    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000

    def __init__(self, classes=None, debug=False):
        super(FasterRCNN, self).__init__()

        ###if classes is not None:
        ###    self.classes = classes
        ###    self.n_classes = len(classes)

        self.rpn = RPN()
        #self.roi_pool = RoIPool(7, 7, 1.0/16)
        #self.fc6 = FC(512 * 7 * 7, 4096)
        #self.fc7 = FC(4096, 4096)
        #self.target_embedding= FC(4096 + 512, 4096)
        #self.score_fc = FC(4096, self.n_classes, relu=False)
        #self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)

        # loss
        self.cross_entropy = None
        self.loss_box = None

        # for log
        self.debug = debug

    @property
    def loss(self):
        # print self.cross_entropy
        # print self.loss_box
        # print self.rpn.cross_entropy
        # print self.rpn.loss_box
        return self.cross_entropy + self.loss_box * 10

    def forward(self, target_data, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        target_features, features, rois,scores = self.rpn(target_data, im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)


        if self.training:
            self.cross_entropy = network.np_to_variable(np.zeros(1))
            self.loss_box = network.np_to_variable(np.zeros(1))

        bbox_pred = network.np_to_variable(np.zeros((rois.size()[0],8)))


        return scores, bbox_pred, rois





















    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return pred_boxes, scores, self.classes[inds]

    def detect(self, image, thr=0.3):
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        cls_prob, bbox_pred, rois = self(im_data, im_info)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
        return pred_boxes, scores, classes

    def load_from_npz(self, params):
        self.rpn.load_from_npz(params)

        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)

