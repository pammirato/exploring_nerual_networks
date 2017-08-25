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
                        'PC_1-5_archC_2_20',
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


    #file names for saving
    base_name = '/playpen/ammirato/Data/Detections/features/'
    feat_name = base_name + name + '_features_train.txt'
    label_name = base_name + name + '_labels_train.txt'




    right = 0
    total = 0
    acc_by_class = np.zeros((2,dataloader.dataset.get_num_classes()))


    all_features = -1*np.ones((num_images*5, 512)) 
    all_labels = -1*np.ones((num_images*5, 1)) 

    
    #for i in range(num_images):
    for il,batch in enumerate(dataloader):

        print '{} / {}'.format(il, len(dataloader))

        # get one batch
        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        gt_boxes = np.asarray(batch[1][0],dtype=np.float32)
 
        # forward
        class_scores, features = net(im_data,gt_boxes)
        #features = net(im_data,gt_boxes)

        features = features.data.cpu().squeeze().numpy()
        probs = torch.nn.functional.softmax(class_scores)
        pred_scores, pred_classes =  torch.topk(class_scores,1)
        pred_scores = torch.nn.functional.softmax(pred_scores)
        pred_classes = pred_classes.data.cpu()
        pred_scores = pred_scores.data.cpu()
        pred_classes = pred_classes.view(33,60).numpy() 
        pred_scores = pred_scores.view(33,60).numpy() 

        _feat_stride = 16
       
        counter = -1 
        #see if each patch in each box was classified correctly or not
        for box in gt_boxes:
            
            true_class = int(box[4])
            #if true_class == 0:
            #    continue
            counter +=1
            box = (box[0:4] -1) / _feat_stride
            box[0] = min(box[0], features.shape[2]-1)
            box[1] = min(box[1], features.shape[1]-1)
            box[2] = min(box[2], features.shape[2]-1)
            box[3] = min(box[3], features.shape[1]-1)
            box = box.astype(np.int32)

            bpc = pred_classes[box[1]:box[3]+1, box[0]:box[2]+1]
            
            all_features[il*5+counter,:] = features[:,box[1],box[0]]
            all_labels[il*5+counter,:] = true_class

            #if len(np.where(bpc == true_class)[0]) > 0:
            #    right +=1
            #    acc_by_class[0,true_class] +=1
            #total+=1
            #acc_by_class[1,true_class] +=1

            right += len(np.where(bpc == true_class)[0])
            acc_by_class[0,true_class] +=len(np.where(bpc == true_class)[0])
            total += bpc.size
            acc_by_class[1,true_class] +=bpc.size

            breakp =1


        #correct = np.where(cls == gt_boxes[4])[0]
    
        #right += correct.shape[0]
        #total += cls.shape[0]

        #acc_by_class[0,gt_boxes[4]] += correct.shape[0]
        #acc_by_class[1,gt_boxes[4]] +=  cls.shape[0]

    acc = float(right)/float(total)
    #print 'acc: {}'.format(acc)
    print 'acc: {}  r: {}  t: {}'.format(acc, right, total)
    print acc_by_class.astype(np.int32)


    bad_inds = np.where(all_labels[:] == -1)[0]
    all_labels = np.delete(all_labels,bad_inds)
    all_features = np.delete(all_features,bad_inds, axis=0)

    np.savetxt(feat_name, all_features)
    np.savetxt(label_name, all_labels)



if __name__ == '__main__':
    # load data
#    imdb = get_imdb(imdb_name)
#    imdb.competition_mode(on=True)
    data_path = '/playpen/ammirato/Data/HalvedRohitData/'
    scene_list=[
             'Home_003_1',
             #'Home_002_1',
             'Home_003_2',
             #'test',
             'Office_001_1'
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


    print 'Dont forget to change chosen object ids to fit model!'


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




