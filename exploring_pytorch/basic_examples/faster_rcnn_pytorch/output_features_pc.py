import os
import torch
import cv2
import cPickle
import numpy as np

from faster_rcnn import network
from faster_rcnn.part_classifier import Part_Classifier 
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD  
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
import exploring_pytorch.basic_examples.GetDataSet as GetDataSet

#import matplotlib.pyplot as plt
import json

# hyper-parameters
# ------------
imdb_name = 'voc_2007_test'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
# trained_model = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
#trained_model = 'models/saved_model3/faster_rcnn_90000.h5'
#trained_model = '/playpen/ammirato/Documents/exploring_neural_networks/exploring_pytorch/saved_models/fasterRCNN_avd.h5'
trained_model_path = ('/playpen/ammirato/Data/Detections/' + 
                     'saved_models/')
trained_model_names = ['fasterRCNN_avd.h5']
trained_model_names = [#'faster_rcnn_avd_split2_instances1-5_0',
#                       'PC_1-5_archA_0_0',
#                       'PC_1-5_archA_0_1',
#                       'PC_1-5_archA_0_2',
#                       'PC_1-5_archA_0_3',
#                       'PC_1-5_archA_0_4',
                       'PC_1-5_archB_0_14',
                       'PC_1-5_archB_0_13',
                       'PC_1-5_archB_0_12',
                       'PC_1-5_archB_0_11',
                       'PC_1-5_archB_0_10',
                       'PC_1-5_archB_0_9',
                       'PC_1-5_archB_0_8',
                       'PC_1-5_archB_0_7',
                       'PC_1-5_archB_0_6',
                       'PC_1-5_archB_0_5',
                       'PC_1-5_archB_0_4',
                       'PC_1-5_archB_0_3',
                       'PC_1-5_archB_0_2',
                       'PC_1-5_archB_0_1',
                       'PC_1-5_archB_0_0',
                      ]
rand_seed = 1024

#save_name = 'faster_rcnn_100000'
max_per_image = 300
thresh = 0.05
vis = False 

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)




target_image = cv2.imread('/playpen/ammirato/Data/advil_liqui_gels_target.jpg')
means = np.array([[[102.9801, 115.9465, 122.7717]]])
target_image = target_image - means
target_image = np.expand_dims(target_image,0)



def test_net(name, net, dataloader, max_per_image=300, thresh=0.05, vis=False,
             output_dir=None):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataloader)

    right = 0
    total = 0
    acc_by_class = np.zeros((2,dataloader.dataset.get_num_classes()))

    #file names for saving
    base_name = '/playpen/ammirato/Data/Detections/features/' 
    feat_name = base_name + model_name + '_features_train.txt'
    label_name = base_name + model_name + '_labels_train.txt'

    #arrays for saving data in
    all_features = -1* np.ones((num_images*2, 512))
    all_labels = -1* np.ones((num_images*2, 1))

    #for i in range(num_images):
    for i,batch in enumerate(dataloader):

        print '{} / {}'.format(i, len(dataloader))

        # get one batch
        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        gt_boxes = np.asarray(batch[1][0],dtype=np.float32)
 
        # forward
        img_features = net(im_data,gt_boxes)
        img_features = img_features.data.cpu().numpy()

        if gt_boxes.shape[0] < 2:
            print 'skip'
            continue 
        gt_inds = np.random.choice(np.arange(gt_boxes.shape[0]),2, replace=False)
        gt_boxes = gt_boxes[[gt_inds[0],gt_inds[1]], :]

        print gt_boxes[:,4]

        #convert boxes to locations in feature map: x,y,id
        gt_boxes[:,0:4] = gt_boxes[:,0:4].astype(np.int32)/16

        gt_boxes[:,0] = np.fmin(gt_boxes[:,0], img_features.shape[3]-1)
        gt_boxes[:,1] = np.fmin(gt_boxes[:,1], img_features.shape[2]-1)
        gt_boxes[:,2] = np.fmin(gt_boxes[:,2], img_features.shape[3]-1)
        gt_boxes[:,3] = np.fmin(gt_boxes[:,3], img_features.shape[2]-1)
        gt_boxes = gt_boxes.astype(np.int32)

        obj1_feats = img_features[0,
                                  :,
                                  gt_boxes[0,1]:gt_boxes[0,3]+1, 
                                  gt_boxes[0,0]:gt_boxes[0,2]+1]
        obj1_feats = np.reshape(obj1_feats,(512, -1))

        obj2_feats = img_features[0,
                                  :,
                                  gt_boxes[1,1]:gt_boxes[1,3]+1, 
                                  gt_boxes[1,0]:gt_boxes[1,2]+1]
        obj2_feats = np.reshape(obj2_feats,(512, -1))

        obj1_ind = np.random.choice(obj1_feats.shape[1])
        obj1_feat = obj1_feats[:,obj1_ind]
        obj2_ind = np.random.choice(obj2_feats.shape[1])
        obj2_feat = obj2_feats[:,obj2_ind]

        all_features[i*2,:] = obj1_feat
        all_features[i*2 + 1,:] = obj1_feat

        all_labels[i*2,:] = gt_boxes[0,4]
        all_labels[i*2 +1,:] = gt_boxes[1,4]


    bad_inds = np.where(all_labels[:] == -1)[0]
    all_labels = np.delete(all_labels,bad_inds)
    all_features = np.delete(all_features,bad_inds, axis=0)

    np.savetxt(feat_name, all_features)
    np.savetxt(label_name, all_labels)


if __name__ == '__main__':
    # load data
    data_path = '/playpen/ammirato/Data/RohitData/'
    scene_list=[
             #'Home_003_1',
             'Home_002_1',
             #'Home_003_2',
             #'test',
             #'Office_001_1'
             ]

    #CREATE TRAIN/TEST splits
    dataset = GetDataSet.get_part_classifier_AVD(data_path,
                                            scene_list,
                                            preload=False,
                                            chosen_ids=range(6),
                                            fraction_of_no_box=.01)

    #create train/test loaders, with CUSTOM COLLATE function
    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              collate_fn=AVD.collate)



    #test multiple trained nets
    for model_name in trained_model_names:
        print model_name
        # load net
        net = Part_Classifier(classes=dataset.get_class_names())
        network.load_net(trained_model_path + model_name+'.h5', net)
        print('load model successfully!')

        net.cuda()
        #net.eval()
        #net.train()

        # evaluation
        test_net(model_name, net, dataloader, max_per_image, thresh=thresh, vis=vis,
                 output_dir='/playpen/ammirato/Data/Detections/FasterRCNN_AVD/')




