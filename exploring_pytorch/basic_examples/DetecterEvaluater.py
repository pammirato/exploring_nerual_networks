import torch
from torch.autograd import Variable
import numpy as np
import json

import GetDataSet as GetDataSet

def get_boxes_iou(box1, box2):
    """
    Returns iou of box1 and  box2
      
    ARGS:
        box1: (numpy array) [xmin, ymin, xmax,ymax]
        box2: (numpy array) [xmin, ymin, xmax,ymax]
    """    

    inter_area =( (min(box1[2],box2[2]) - max(box1[0],box2[0])) *
                  (min(box1[3],box2[3]) - max(box1[1],box2[1])))

    if(inter_area<0):
        return 0

    union_area = ((box1[2]-box1[0])*(box1[3]-box1[1]) + 
                  (box2[2]-box2[0])*(box2[3]-box2[1]) -
                  inter_area)

    return inter_area/union_area 





class  DetectorEvaluater(object):
    """
    Calculates MAP for a given detector and dataset 
    """

    def __init__(self,
                 score_thresholds=np.linspace(0,1,101),
                 recall_thresholds = np.linspace(0,1,11),
                 iou_threshold=.5): 
        """
        Initializes evaluater

        KWARGS:
            score_thresholds(numpy array= np.linspace(0,1,101)): 
                                    thresholds to use for each
                                    precision/recall point
            recall_thresholds(numpy array= np.linspace(0,1,11)): 
                                    thresholds to use for computing 
                                    PASCAL VOC style average precision
            iou_threshold(float = .5): min iou for correct detection
        """

        self.score_thresholds = score_thresholds
        self.recall_thresholds = recall_thresholds
        self.iou_threshold = iou_threshold
        self.min_iou_for_loc_error = .3


    def run(self, det_results, gt_labels,class_ids, max_difficulty=4):
        """
        Runs evaluation given detection results and ground truth

        ex) evaluater.run(det_results, gt_labels,class_ids)

        ARGS:
            det_results: dict from image_name to bboxes 
            gt_labels: dict from image_name to bboxes 
            class_ids(List[int,...]): list of class ids
        KARGS:
            max_difficulty (int): the max difficulty of box to consider
        """
        RIGHT = 0
        WRONG = 1
        #TODO: -allow multiple gt boxes for same class in same image
        #      -don't count harder boxes as wrong       


        #for diagnosing error:
        #Recall:
            #missed - no box, regardless of class, has iou with gt_box
        #Precision:
            #nms - this box was good, but object already detected        
            #loc - intersects some of gt but not enough
            #con - box is not correct, and it has iou with a gt_box of another class 
            #bg - box is not correct, and it does not have iou with a gt_box of another class 
        #Are detectors independent?? matters for above
        missed_errors = {} 
        loc_errors = {} 
        con_errors = {} 
        bg_errors = {} 
        nms_errors = {} 


        #dicts from class id to num right/wrong, total num of gt boxes
        right_wrong = {}
        gt_total = {}
        for cid in class_ids:
            #num right/wrong for each score thresh
            right_wrong[cid] = np.zeros((len(self.score_thresholds),2))
            gt_total[cid] = 0

            missed_errors[cid] = 0 
            loc_errors[cid] = 0 
            con_errors[cid] = 0 
            bg_errors[cid] = 0 
            nms_errors[cid] = 0



        #get all the image names
        all_image_names = gt_labels.keys()

        #for each image 
        for image_name in all_image_names: 

            #get the gt and detected boxes            
            gt_boxes = np.asarray(gt_labels[image_name])
            #if gt_boxes.shape[0] != 0:
            #    gt_boxes = gt_boxes[np.where(gt_boxes[:,5]<=max_difficulty)[0],:]
            if gt_boxes.shape[0] == 0:
                gt_boxes = np.zeros((0,6))
            try:
                det_boxes = np.asarray(det_results[image_name])
            except:
                continue
            if det_boxes.shape[0] == 0:
                det_boxes = np.zeros((0,6))

            #get ious of all gt_boxes with all det_boxes
            gt_boxes_iou = np.zeros((gt_boxes.shape[0], det_boxes.shape[0]))
            for il,gt_box in enumerate(gt_boxes):
                if not gt_box[4] in class_ids:#we are not considering this class
                    continue
                for jl,det_box in enumerate(det_boxes):
                    gt_boxes_iou[il,jl] = get_boxes_iou(gt_box,det_box) 
                #mark if this gt box has no iou with any det_box
                iou_inds = np.where(gt_boxes_iou[il,:] > self.min_iou_for_loc_error)[0]
                if iou_inds.shape[0] == 0:
                    missed_errors[gt_box[4]] += 1  


            breakp = 1    
             
            #for each class id, get num right/wrong/gt
            for cid in class_ids: 
                #get gt and detected boxes for this class
                gt_inds = np.where(gt_boxes[:,4] == cid)[0]
                gt_class_box = gt_boxes[gt_inds,:]
                det_inds = np.where(det_boxes[:,4] == cid)[0]
                det_class_boxes = det_boxes[det_inds,:]

                #count gt boxes, keep track of which are detected
                if gt_class_box.shape[0] > 0 and gt_class_box[0,5] <= max_difficulty:
                    gt_total[cid] += gt_class_box.shape[0]
                gt_box_detected = np.zeros(gt_class_box.shape[0])

                #sort detected boxes, so we check higher scoring
                #boxes first
                det_class_boxes = det_class_boxes[det_class_boxes[:,5].argsort()][::-1]

                #see if each detected box is correct or not
                for box_ind, box in enumerate(det_class_boxes):
                    correct = False
                    if gt_class_box.shape[0] > 0:
                        #get boxes iou with gt_boxes
                        iou = get_boxes_iou(box,gt_class_box[0,:]) 
                        #see if iou passes threshold for the gt_box
                        if iou > self.iou_threshold: 
                            #see if  gt_box is already detected
                            if not gt_box_detected[0]:
                                correct = True 
                                #mark gt_box as detected
                                gt_box_detected[0] = 1
                            else:#this box should have been supressed?
                                nms_errors[cid] += 1
                        else:#box did not have iou with gt box of this class
                            if iou > self.min_iou_for_loc_error:
                                #it had some iou, just not enough
                                loc_errors[cid] += 1 
                            else:#it really was not near the gt box
                                #did if have iou with another gt_box?
                                #get this boxes iou with all gt boxes
                                boxes_iou = gt_boxes_iou[:,box_ind]
                                iou_inds = np.where(boxes_iou > self.iou_threshold)[0]
                                if iou_inds.shape[0] > 0:
                                    con_errors[cid] += 1
                                else:
                                    bg_errors[cid] += 1                                    
                            
                    #mark correct/wrong for each score threshold
                    for il,score_thresh in enumerate(self.score_thresholds):
                        if box[5] > score_thresh: 
                            if correct:
                                if gt_class_box[0,5] <= max_difficulty:
                                    right_wrong[cid][il,RIGHT] += 1
                            else: 
                                right_wrong[cid][il,WRONG] += 1

        self.right_wrong = right_wrong
        self.gt_total = gt_total 


 
        #now compute precision/recall for each class, score threshold
        precision_recall = np.zeros((len(class_ids),
                                    len(self.score_thresholds),
                                    2))
        P = 0
        R = 1
        #average precision per class
        avg_prec = np.zeros(len(class_ids))
        #for plotting precision recall curve
        all_max_precisions = np.zeros((len(class_ids),
                                       self.recall_thresholds.shape[0]))

       
        for il,cid in enumerate(class_ids):
            #skip classes with no gt_boxes
            if gt_total[cid] == 0:
                #flag ap for this class as invalid
                avg_prec[cid] = -1
                continue
    
             
            for jl in range(len(self.score_thresholds)):             
                #skip thresholds where no detections happen
                #recall is 0, precision undefined
                if(right_wrong[cid][jl,RIGHT] == 0 and
                   right_wrong[cid][jl,WRONG]==0):
                    continue
                #precision = num_right / (num_right+num_wrong)
                precision_recall[il,jl,P] = (right_wrong[cid][jl,RIGHT]/
                                              (right_wrong[cid][jl,RIGHT] + 
                                               right_wrong[cid][jl,WRONG]))
                #recall = num_right / gt_total
                precision_recall[il,jl,R] = (right_wrong[cid][jl,RIGHT]/
                                              gt_total[cid])

            ##compute average precision
            
            #get the max preicsion at each of these recalls 
            max_precisions = np.zeros(self.recall_thresholds.shape[0])
            for jl,recall in enumerate(self.recall_thresholds):
                #find the max precision at this recall (or greater)
                max_prec = 0
                for kl in range(precision_recall.shape[1]):
                    prec = precision_recall[il,kl,P] 
                    rec = precision_recall[il,kl,R] 
                    #recall must be at least target recall
                    if rec >= recall:
                        if prec > max_prec:
                            max_prec = prec 
                max_precisions[jl] = max_prec

            avg_prec[il] = max_precisions.mean() 
            all_max_precisions[il] = max_precisions


        #consolidate errors
        all_errors = {}
        all_errors['loc'] = loc_errors
        all_errors['nms'] = nms_errors
        all_errors['missed'] = missed_errors 
        all_errors['con'] = con_errors 
        all_errors['bg'] = bg_errors 

        #return  map
        mean_ap = avg_prec[np.where(avg_prec>=0)].mean()
        return [mean_ap, avg_prec, all_max_precisions, all_errors]















#    def run_with_detector(self,detector, dataloader,
#                          cuda=False):
#        """
#        Runs the evaluation, while running the detector on each image
#
#        detector (torch.nn.Module or similar): assume on cpu
#                        output should be list of lists
#                        [[xmin,xmax,ymin,ymax,id,score],...]
#        dataloader (torch.utils.data.DataLoader): assumes batch_size=1
#        KWARGS:
#            cuda(bool = False): whether to run model on GPU
#        """
#        #TODO: run batches of images
#        #TODO: make more effecient with matrix operations(less loops) 
#
#        #for each class, for each score threshold, keep track of  
#        #num right/wrong
#        class_ids = [i for i in range(dataloader.get_num_classes())]
#        right_wrong = np.zeros((len(class_ids),
#                                    len(self.score_thresholds),
#                                    2))
#        #keep track of how many boxes of each class there are in ground truth
#        gt_total = np.zeros(len(class_ids))
#
#        #for each image, get detector output, and record 
#        #preicsion/recall scores 
#        for il,data in enumerate(dataloader):
#            #get image,box labels and put on gpu if needed
#            img,labels = data 
#            labels = labels[0]#get boxes(not movements)
#            if cuda:
#                img,labels = img.cuda(), labels.cuda()
#                detector.cuda()
#
#            #run image through detector
#            det_out = detector(img) 
#
#            labels = labels.cpu().numpy()
#            dets_out = np.asarray(dets_out)
#
#            #record each ground truth box
#            for box in labels:
#                gt_total[box[4]] +=1
#        
#            #record each detected box as right or wrong
#            #TODO - make more effecient
#            for box in det_out:
#                correct = False 
#                #check if a gt_box with the same label exists
#                for gt_box in labels:
#                    if gt_box[4] == box[4]:
#                        iou = get_boxes_iou(box,gt_box)
#                        if iou>self.iou_threshold:
#                            correct = True
#
#                t_ind = 0
#                for thresh in self.score_thresholds:
#                    if box[5] >= thresh:
#                        if correct:
#                            right_wrong[box[4],t_ind,0] +=1
#                        else:
#                            right_wrong[box[4],t_ind,1] +=1
#                    else:
#                        right_wrong[box[4],t_ind,1] +=1
#
#
#                    t_ind += 1
#                            
#                
#        #for debugging 
#        self.right_wrong = right_wrong 
#        self.gt_totat = gt_total            
# 
#        #now compute precision/recall for each score threshold
#        precision_recall = np.zeros((len(class_ids),
#                                    len(self.score_thresholds),
#                                    2))
#        for il in range(len(class_ids)):
#            for jl in range(len(self.score_thresholds)):             
#                precision_recall[il,jl,0] = (right_wrong[il,jl,0]/
#                                              (right_wrong[il,jl,0] + 
#                                               right_wrong[il,jl,1])) 
#
#                precision_recall[il,jl,1] = (right_wrong[il,jl,0]/
#                                              gt_total[il])
#        #put model back on cpu
#        if cuda:
#            detector.cpu()
#
#        return precision_recall





if __name__ == '__main__':

    #set up dataset to get ground truth(gt) data
    data_path = '/playpen/ammirato/Data/HalvedRohitData/'
    scene_list=[
             'Home_006_1',
             'Home_008_1',
             #'test',
             'Home_002_1'
             ] 

    max_difficulty=4
    dataset = GetDataSet.get_fasterRCNN_AVD(data_path,
                                            scene_list,
                                            chosen_ids=range(0,28))#[0,1,2,3,4,5])
    gt_boxes = dataset.get_original_bboxes()


    #get outputs from testing model
    trained_model_names = ['faster_rcnn_avd_split2_target_driven_fc7+_concat_0',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_1',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_2',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_3',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_4',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_5',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_6',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_7',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_8',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_9',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_10',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_11',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_12',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_13',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_14',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_15',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_16',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_17',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_18',
                           'faster_rcnn_avd_split2_target_driven_fc7+_concat_19',
                          ]

    for model_name in trained_model_names:

        with open('/playpen/ammirato/Data/Detections/' + 
                  'FasterRCNN_AVD/target_driven/' + model_name+'.json') as f:
            det_results = json.load(f)

        evaluater = DetectorEvaluater(score_thresholds=np.linspace(0,1,11),
                                      recall_thresholds=np.linspace(0,1,11))
        m_ap,ap,max_p,errors = evaluater.run(det_results,gt_boxes,[1,2,3,4,5],
                                             max_difficulty=max_difficulty)
        #print 'MAPish {}'.format(ap[ap.nonzero()].mean())
        print 'MAPish {}'.format(m_ap)