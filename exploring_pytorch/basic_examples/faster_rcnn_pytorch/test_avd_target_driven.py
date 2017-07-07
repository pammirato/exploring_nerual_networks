import os
import torch
import cv2
import cPickle
import numpy as np

from faster_rcnn import network
from faster_rcnn.faster_rcnn_target_driven import FasterRCNN, RPN
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


trained_model_names=[#'faster_rcnn_avd_split2_target_driven_fc7+_concat_vgg_feat_concat_train7_19',
                    'FRA_TD_1-5_archA_2_50',
                    'FRA_TD_1-5_archA_2_40',
                    'FRA_TD_1-5_archA_2_30',
                    'FRA_TD_1-5_archA_2_20',
                    'FRA_TD_1-5_archA_2_10',
                    'FRA_TD_1-5_archA_2_45',
                    'FRA_TD_1-5_archA_2_0',
                    #'FRA_TD_1-5_archB_30',
                    #'FRA_TD_1-5_archB_10',
                    #'FRA_TD_1-5_archB_20',
                    #'FRA_TD_1-5_archB_25',
                    #'FRA_TD_1-5_archB_15',
                    #'FRA_TD_1-5_archB_5',
                    ]
rand_seed = 1024

#save_name = 'faster_rcnn_100000'
max_per_image = 1 
thresh = 0.05
vis = False 

#load all target images
target_path = '/playpen/ammirato/Data/big_bird_crops_160'
image_names = os.listdir(target_path)
image_names.sort()
target_images = []
for il, name in enumerate(image_names):
    if il >4:
        continue
    target_data = cv2.imread(os.path.join(target_path,name))
    target_data = np.expand_dims(target_data,axis=0)
    target_images.append(target_data)






# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


#def im_detect(net, image):
def im_detect(net, target_data, im_data, im_info):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    #im_data, im_scales = net.get_image_blob(image)
    #im_info = np.array(
    #    [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
    #    dtype=np.float32)

    cls_prob, bbox_pred, rois = net(target_data, im_data, im_info)
    scores = cls_prob.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        #pred_boxes = clip_boxes(pred_boxes, image.shape)
        pred_boxes = clip_boxes(pred_boxes, im_data.shape[1:])
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def test_net(name, net, dataloader, max_per_image=300, thresh=0.05, vis=False,
             output_dir=None):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataloader)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(dataloader.dataset.get_num_classes())]
    #array of result dicts
    all_results = {} 
    #output_dir = get_output_dir(imdb, name)
    

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    
    if output_dir is not None:
        #det_file = os.path.join(output_dir, 'detections.pkl')
        det_file = os.path.join(output_dir, name+'.json')
        print det_file

    #for i in range(num_images):
    for i,batch in enumerate(dataloader):
        #im = cv2.imread(imdb.image_path_at(i))

        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        im_info = np.zeros((1,3))
        im_info[0,:] = [im_data.shape[1],im_data.shape[2],1]
        dontcare_areas = np.zeros((0,4))       

        im = im_data.squeeze()
        im = im.copy()
        means = np.array([[[102.9801, 115.9465, 122.7717]]])
        im -= means
        im = im.astype(np.uint8)
 
        _t['im_detect'].tic()

        all_image_dets = np.zeros((0,6)) 
        for j,target_data in enumerate(target_images):
        #scores, boxes = im_detect(net, im)
            scores, boxes = im_detect(net, target_data, im_data, im_info)
        

            detect_time = _t['im_detect'].toc(average=False)

            _t['misc'].tic()
            if vis:
                # im2show = np.copy(im[:, :, (2, 1, 0)])
                im2show = np.copy(im)

            #separate boxes by class, non maximum supression
            # skip j = 0, because it's the background class
            inds = np.where(scores[:, 1] > thresh)[0]
            cls_scores = scores[inds, 1]
            cls_boxes = boxes[inds, 1 * 4:(1 + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]

                    
            if vis:
                #im2show = vis_detections(im2show, imdb.classes[j], cls_dets)
                im2show = vis_detections(im2show, dataloader.dataset.class_id_to_name[j],
                                         cls_dets)
            all_boxes[j][i] = cls_dets

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
            nms_time = _t['misc'].toc(average=False)

            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                .format(i + 1, num_images, detect_time, nms_time)

            if vis:
                #cv2.imshow('test', im2show)
                #cv2.waitKey(1)
                #plt.imshow(im2show)
                #plt.draw()
                #plt.pause(.001)
                raw_input('press enter')

            #make a list of all detections in this image
            class_dets = all_boxes[j][i]
            #put class id in the box
            class_dets = np.insert(class_dets,4,j+1,axis=1)
            all_image_dets = np.vstack((all_image_dets,class_dets))
            #for box in class_dets:
            #    result = {'image_name':batch[1][1],
            #              'instance_id': j,
            #              'bbox':box[0:4],
            #              'score':box[4]
            #             }
            #    all_results.append(result)
            #result = {'image_name':batch[1][1],
            #          'instance_id': j,
            #          'bboxes':class_dets.to_list(),
            #          'score':box[4]
            #         }
            #all_results.append(result)
        #all_results[batch[1][1]] = all_image_dets.astype(np.int32).tolist()
        all_results[batch[1][1]] = all_image_dets.tolist()
    if output_dir is not None:
        #with open(det_file, 'wb') as f:
        #    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
        with open(det_file, 'w') as f:
            json.dump(all_results,f)
   # print 'Evaluating detections'
   # imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    # load data
#    imdb = get_imdb(imdb_name)
#    imdb.competition_mode(on=True)
    data_path = '/playpen/ammirato/Data/HalvedRohitData/'
    scene_list=[
             'Home_003_1',
             'Home_003_2',
             #'test',
             'Office_001_1'
             ]

    #CREATE TRAIN/TEST splits
    dataset = GetDataSet.get_fasterRCNN_AVD(data_path,
                                            scene_list,
                                            preload=False,
                                            chosen_ids=[1,2,3,4,5], 
                                            by_box=False,
                                            fraction_of_no_box=1)

    #create train/test loaders, with CUSTOM COLLATE function
    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              collate_fn=AVD.collate)



    #test multiple trained nets
    for model_name in trained_model_names:
        print model_name
        # load net
        #net = FasterRCNN(classes=imdb.classes, debug=False)
        net = FasterRCNN(classes=dataset.get_class_names(), debug=False)
        network.load_net(trained_model_path + model_name+'.h5', net)
        print('load model successfully!')

        net.cuda()
        net.eval()

        # evaluation
        test_net(model_name, net, dataloader, max_per_image, thresh=thresh, vis=vis,
                 output_dir='/playpen/ammirato/Data/Detections/FasterRCNN_AVD/')




