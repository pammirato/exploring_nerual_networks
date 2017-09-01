import os
import torch
import cv2
import cPickle
import numpy as np

from faster_rcnn import network
from faster_rcnn.patch_detector import Patch_Detector
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
                        'PD_1-5_archA_0_9',
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


data_path = '/playpen/ammirato/Data/HalvedRohitData/'
id_to_name = GetDataSet.get_class_id_to_name_dict(data_path)
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
    acc_by_class = np.zeros((4,dataloader.dataset.get_num_classes()))


    all_features = -1*np.ones((num_images*5, 512)) 
    all_labels = -1*np.ones((num_images*5, 1)) 

    
    #for i in range(num_images):
    for il,batch in enumerate(dataloader):

        print '{} / {}'.format(il, len(dataloader))

        # get one batch
        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        gt_boxes = np.asarray(batch[1][0],dtype=np.float32)
 

        _feat_stride = 16
       
        counter = -1 
        #see if each patch in each box was classified correctly or not
        for il,box in enumerate(gt_boxes):
            
            true_class = int(box[4])
            if true_class == 0:
                continue
            counter +=1
            box = (box[0:4] -1) / _feat_stride
            box[0] = min(box[0], 60-1)
            box[1] = min(box[1], 33-1)
            box[2] = min(box[2], 60-1)
            box[3] = min(box[3], 33-1)
            box = box.astype(np.int32)

            target_img = target_images[id_to_name[true_class]]
           
            # forward
            heat_map = net(im_data,target_img)
            heat_map = heat_map.data.cpu().numpy().squeeze()
            bpc = heat_map[box[1]:box[3]+1, box[0]:box[2]+1]
 
            right += len(np.where(bpc > 300)[0])
            acc_by_class[0,true_class] += (len(np.where(bpc > 300)[0]) > 0)
            total += bpc.size
            acc_by_class[1,true_class] += 1 #bpc.size

           


            #check other gt boxes
            other_boxes = np.delete(gt_boxes,il,0)
            if other_boxes.shape[0] > 0:
                for obox in other_boxes:
                    if obox[4] == 0:
                        continue
                    obox = (obox[0:4] -1) / _feat_stride
                    obox[0] = min(obox[0], 60-1)
                    obox[1] = min(obox[1], 33-1)
                    obox[2] = min(obox[2], 60-1)
                    obox[3] = min(obox[3], 33-1)
                    obox = obox.astype(np.int32)
                    obpc = heat_map[obox[1]:obox[3]+1, obox[0]:obox[2]+1]
                    right += len(np.where(obpc < 300)[0])
                    acc_by_class[2,true_class] += (len(np.where(obpc < 300)[0]) > 0)
                    total += obpc.size
                    acc_by_class[3,true_class] += 1 #bpc.size



 
            #get a random bg box
            xmin = np.random.randint(0,58) 
            ymin = np.random.randint(0,31) 
            width = np.random.randint(1,10) 
            height = np.random.randint(1,10)
            xmax = min(59,xmin+width)
            ymax = min(32,ymin+height)
    
    
            #check to see if the background box intersects any object box
            good_box = True
            if not(xmax<box[0] or ymax<box[1] or xmin>box[2] or ymin>box[3]):
                good_box = False

            if good_box:
                true_class = 0
                box = [xmin, ymin, xmax,ymax]
                bpc = heat_map[box[1]:box[3]+1, box[0]:box[2]+1]
     
                right += len(np.where(bpc < 300)[0])
                acc_by_class[0,true_class] += (len(np.where(bpc < 300)[0]) > 0)
                total += bpc.size
                acc_by_class[1,true_class] += 1 #bpc.size


                            


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

    #np.savetxt(feat_name, all_features)
    #np.savetxt(label_name, all_labels)



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
        net = Patch_Detector(classes=dataset.get_class_names())
        network.load_net(trained_model_path + model_name+'.h5', net)
        print('load model successfully!')

        net.cuda()
        #net.eval()
        #net.train()

        # evaluation
        test_net(model_name, net, dataloader, max_per_image, thresh=thresh, vis=vis,
                 output_dir='/playpen/ammirato/Data/Detections/FasterRCNN_AVD/')




