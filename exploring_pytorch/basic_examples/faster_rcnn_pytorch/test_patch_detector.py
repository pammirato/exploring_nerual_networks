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
                        'PD_1-5_archA_4_2',
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
name_to_id = {}
for cid in id_to_name.keys():
    name_to_id[id_to_name[cid]] = cid

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

    if output_dir is not None:
        #det_file = os.path.join(output_dir, 'detections.pkl')
        det_file = os.path.join(output_dir, name+'.json')
        print det_file

    all_results = {}
    
    #for i in range(num_images):
    for il,batch in enumerate(dataloader):
        print '{} / {}'.format(il, len(dataloader))

        all_image_dets = np.zeros((0,6))

        # get one batch
        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        gt_boxes = np.asarray(batch[1][0],dtype=np.float32)
 

        _feat_stride = 16
       
        counter = -1 
        #see if each patch in each box was classified correctly or not
        for jl,target_name in enumerate(dataloader.dataset.get_class_names()): 

            if target_name == 'background':
                continue
            target_data = target_images[target_name]
            jl = int(name_to_id[target_name]-1)

            # forward
            heat_map = net(im_data,target_data)
            heat_map = heat_map.data.cpu().numpy().squeeze()

            max_pos = np.where(heat_map > 450)

            if max_pos[1].shape[0] < 1:
                continue

            cur_det = [max_pos[1].min(), max_pos[0].min(),
                       max_pos[1].max(), max_pos[0].max(), 0]
                       
            class_dets = np.expand_dims(np.array(cur_det)*16,0)
            class_dets[0,-1] = heat_map.max() 


            class_dets = np.insert(class_dets,4,jl+1,axis=1)
            all_image_dets = np.vstack((all_image_dets,class_dets))

        all_results[batch[1][1]] = all_image_dets.tolist()
    if output_dir is not None:
        with open(det_file, 'w') as f:
            json.dump(all_results,f)




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




