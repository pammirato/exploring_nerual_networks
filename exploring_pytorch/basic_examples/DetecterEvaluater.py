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

    inter_width =min(box1[2],box2[2]) - max(box1[0],box2[0])
    inter_height = min(box1[3],box2[3]) - max(box1[1],box2[1])
    inter_area =  inter_width*inter_height
                  

    if(inter_width<0 or inter_height<0):
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
                 iou_thresholds=np.array([.5,.1,.25,.75])): 
        """
        Initializes evaluater

        KWARGS:
            score_thresholds(numpy array= np.linspace(0,1,101)): 
                                    thresholds to use for each
                                    precision/recall point
            recall_thresholds(numpy array= np.linspace(0,1,11)): 
                                    thresholds to use for computing 
                                    PASCAL VOC style average precision
            iou_thresholds(numpy array = [.5,.1.,.25,.75):
                                     min iou for correct detection. First
                                     element is the 'main' threshold. This
                                     is the threshold used for error analysis.
        """

        self.score_thresholds = score_thresholds
        self.recall_thresholds = recall_thresholds
        self.iou_thresholds = iou_thresholds
        self.min_iou_for_loc_error = .3


    def run(self, det_results, gt_labels,class_ids,
            max_difficulty=4,
            difficulty_classifier=None):
        """
        Runs evaluation given detection results and ground truth

        ex) evaluater.run(det_results, gt_labels,class_ids)

        ARGS:
            det_results: dict from image_name to bboxes 
            gt_labels: dict from image_name to bboxes 
            class_ids(List[int,...]): list of class ids
        KARGS:
            max_difficulty (int): the max difficulty of box to consider
            difficulty_classifier (None): function that will classify box's 
                                          difficulty
        """
        #TODO: -allow multiple gt boxes for same class in same image
        #      -don't count harder boxes as wrong       

        #some definitions
        RIGHT = 0
        WRONG = 1
        NO_ERR = 0 
        MISS_ERR = 1 
        LOC_ERR = 2 
        CON_ERR = 3 
        BG_ERR = 4 
        NMS_ERR = 5 
       
        #what difficulties to consider 
        valid_diffs = np.arange(1,max_difficulty+1)

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
        correct_cnt = {}

        scored_errors = {}

        no_boxes_outputted = 0 
        skipped_images = 0

        #dicts from class id to num right/wrong, total num of gt boxes
        right_wrong = {}
        gt_total = {}
        for cid in class_ids:

            #count each type of error for each score threshold
            scored_errors[cid] = np.zeros((6,len(self.score_thresholds)))
    
            #num right/wrong for each score thresh, for each difficulty
            right_wrong[cid] = np.zeros((len(self.iou_thresholds),
                                         max_difficulty+1,
                                         len(self.score_thresholds),
                                         2))
            gt_total[cid] =  np.zeros(max_difficulty+1)
            missed_errors[cid] = 0 
            loc_errors[cid] = 0 
            con_errors[cid] = 0 
            bg_errors[cid] = 0 
            nms_errors[cid] = 0
            correct_cnt[cid] = 0


        #get all the image names
        all_image_names = gt_labels.keys()

        gt_names = []
        no_names = []

        #for each image 
        for image_name in all_image_names: 

            if image_name == '000310007590101.jpg':
                breakp = 1

            #get the gt and detected boxes            
            gt_boxes = np.asarray(gt_labels[image_name])
            if gt_boxes.shape[0] == 0:
                gt_boxes = np.zeros((0,6))
            try:
                det_boxes = np.asarray(det_results[image_name])
            except:
                #print 'No detections for this image!'
                skipped_images +=1 
                continue
            if det_boxes.shape[0] == 0:
                det_boxes = np.zeros((0,6))
                no_boxes_outputted += 1

            #get ious of all gt_boxes with all det_boxes
            gt_boxes_iou = np.zeros((gt_boxes.shape[0], det_boxes.shape[0]))
            for il,gt_box in enumerate(gt_boxes):
                if not gt_box[4] in class_ids:#we are not considering this class
                    continue
                for jl,det_box in enumerate(det_boxes):
                    gt_boxes_iou[il,jl] = get_boxes_iou(gt_box,det_box) 
                #mark if this gt box has no iou with any det_box
                iou_inds = np.where(gt_boxes_iou[il,:] > self.min_iou_for_loc_error)[0]
                if iou_inds.shape[0] == 0 and gt_box[5] <= max_difficulty:
                    missed_errors[gt_box[4]] += 1  
                    scored_errors[gt_box[4]][MISS_ERR,:] +=1

            breakp = 1    
             
            #for each class id, get num right/wrong/gt
            for cid in class_ids: 
                #get gt and detected boxes for this class
                gt_inds = np.where(gt_boxes[:,4] == cid)[0]
                gt_class_box = gt_boxes[gt_inds,:]
                det_inds = np.where(det_boxes[:,4] == cid)[0]
                det_class_boxes = det_boxes[det_inds,:]

                if gt_class_box.shape[0] >0 and cid == 1:
                    breakp=1



                #get gt box difficulty if not defined
                if gt_class_box.shape[0] > 0 and gt_class_box[0,5] not in valid_diffs:
                    #print '{} {}'.format(cid, gt_class_box[0,5])
                    if difficulty_classifier is None:
                        gt_class_box[0,5] = 0
                    else:
                        gt_class_box[0,5] = difficulty_classifier(gt_class_box[0,:]*2.0) 

                #count gt boxes, keep track of which are detected
                if gt_class_box.shape[0] > 0 and gt_class_box[0,5] <= max_difficulty:
                    gt_total[cid][0] += gt_class_box.shape[0]
                    gt_total[cid][gt_class_box[0,5]] += gt_class_box.shape[0]
                    if cid == 2:
                        gt_names.append(image_name)
                
                gt_box_detected = np.zeros(gt_class_box.shape[0])

                #sort detected boxes, so we check higher scoring
                #boxes first
                det_class_boxes = det_class_boxes[det_class_boxes[:,5].argsort()][::-1]


                #see if each detected box is correct or not
                error_type = -1 
                box_correct = np.zeros(len(self.iou_thresholds))
                for box_ind, box in enumerate(det_class_boxes):
                    correct = False
                    if gt_class_box.shape[0] > 0:
                        #get boxes iou with gt_boxes
                        iou = get_boxes_iou(box,gt_class_box[0,:]) 

                        #check for right/wrong for all iou_thresholds
                        for il,iou_thresh in enumerate(self.iou_thresholds):
                            if iou >= iou_thresh and not gt_box_detected[0]:
                               box_correct[il] = 1 

                        #do error analysis for just the main iou_threshold
                        #see if iou passes threshold for the gt_box
                        if iou >= self.iou_thresholds[0]: 
                            #see if  gt_box is already detected
                            if not gt_box_detected[0]:
                                correct = True 
                                correct_cnt[cid] += 1
                                #if cid == 2 and gt_class_box[0,5] <= max_difficulty:
                                #    no_names.append(image_name)
                                #mark gt_box as detected
                                gt_box_detected[0] = 1
                                if gt_class_box[0,5] <= max_difficulty:
                                    error_type = NO_ERR 
                            else:#this box should have been supressed?
                                nms_errors[cid] += 1
                                error_type = NMS_ERR
                        else:#box did not have iou with gt box of this class
                            if iou > self.min_iou_for_loc_error:
                                #it had some iou, just not enough
                                loc_errors[cid] += 1
                                error_type = LOC_ERR 
                            else:#it really was not near the gt box
                                #did if have iou with another gt_box?
                                #get this boxes iou with all gt boxes
                                boxes_iou = gt_boxes_iou[:,box_ind]
                                iou_inds = np.where(boxes_iou > self.iou_thresholds[0])[0]
                                #iou_inds = np.where(boxes_iou > 0)[0]
                                if iou_inds.shape[0] > 0:
                                    con_errors[cid] += 1
                                    error_type = CON_ERR 
                                else:
                                    bg_errors[cid] += 1                                    
                                    error_type = BG_ERR 
                    else:#there was no gt box for this class
                        #did if have iou with another gt_box?
                        #get this boxes iou with all gt boxes
                        boxes_iou = gt_boxes_iou[:,box_ind]
                        iou_inds = np.where(boxes_iou > self.iou_thresholds[0])[0]
                        #iou_inds = np.where(boxes_iou > 0)[0]
                        if iou_inds.shape[0] > 0:
                            con_errors[cid] += 1
                            error_type = CON_ERR 
                        else:
                            bg_errors[cid] += 1                                    
                            error_type = BG_ERR 
                    
                    #classify detected box        
                    box_diff = difficulty_classifier(box*2.0)
                    if box_diff not in valid_diffs:
                        box_diff = 0

                    #mark correct/wrong for each score threshold
                    for il,score_thresh in enumerate(self.score_thresholds):
                        if box[5] >= score_thresh: 
                            for iou_ind,iou_thresh in enumerate(self.iou_thresholds):
                                if box_correct[iou_ind]:
                                    if gt_class_box[0,5] <= max_difficulty:
                                        right_wrong[cid][iou_ind,gt_class_box[0,5],il,0] += 1
                                else:#box was wrong 
                                    right_wrong[cid][iou_ind,box_diff,il,1] += 1

                            #record the error type, just for main iou thresh
                            scored_errors[cid][error_type,il] += 1 




 
        #now compute precision/recall for each class, score threshold
        precision_recall = np.zeros((len(self.iou_thresholds),
                                    max_difficulty+1,
                                    len(class_ids),
                                    len(self.score_thresholds),
                                    2))
        P = 0
        R = 1
        #average precision per class
        avg_prec = np.zeros((len(self.iou_thresholds),
                             max_difficulty+1,
                             len(class_ids)))
        #for plotting precision recall curve
        all_max_precisions = np.zeros((len(self.iou_thresholds),
                                       max_difficulty+1,
                                       len(class_ids),
                                       self.recall_thresholds.shape[0]))

       
        for il,cid in enumerate(class_ids):
            #skip classes with no gt_boxes
            if gt_total[cid][0] == 0:
                #flag ap for this class as invalid
                avg_prec[:,:,cid] = -1
                continue
    
             
            for jl in range(len(self.score_thresholds)):             

                for iou_ind in range(len(self.iou_thresholds)):
                    #skip thresholds where no detections happen
                    #recall is 0, precision undefined
                    if(np.sum(right_wrong[cid][iou_ind,:,jl,:]) == 0 ):# and
                        continue
                    precision_recall[iou_ind,0,il,jl,P] = \
                                                (np.sum(right_wrong[cid][iou_ind,:,jl,0])/
                                                 (np.sum(right_wrong[cid][iou_ind,:,jl,0]) + 
                                                  np.sum(right_wrong[cid][iou_ind,:,jl,1])))
                    #recall = num_right / gt_total
                    precision_recall[iou_ind,0,il,jl,R] = \
                                                 (np.sum(right_wrong[cid][iou_ind,:,jl,0])/
                                                  gt_total[cid][0])


                    #get precision/recall for each difficulty
                    for diff in valid_diffs:
                        #same as above
                        if(np.sum(right_wrong[cid][iou_ind,diff,jl,:]) == 0 or 
                           gt_total[cid][diff] == 0):
                            continue
                        precision_recall[iou_ind,diff,il,jl,P] = \
                                               (np.sum(right_wrong[cid][iou_ind,diff,jl,0])/
                                                (np.sum(right_wrong[cid][iou_ind,diff,jl,0]) + 
                                                 np.sum(right_wrong[cid][iou_ind,diff,jl,1])))
                        precision_recall[iou_ind,diff,il,jl,R] = \
                                               (np.sum(right_wrong[cid][iou_ind,diff,jl,0])/
                                                gt_total[cid][diff])
                    

            ##compute average precision
            
            #get the max preicsion at each of these recalls 
            max_precisions = np.zeros((len(self.iou_thresholds),
                                       max_difficulty+1,
                                       self.recall_thresholds.shape[0]))
            for jl,recall in enumerate(self.recall_thresholds):
                for iou_ind in range(len(self.iou_thresholds)):
                    #find the max precision at this recall (or greater)
                    max_prec = np.zeros(max_difficulty+1)
                    for kl in range(precision_recall.shape[3]):#for each score thresh
                        for ll in range(max_difficulty+1):
                            prec = precision_recall[iou_ind,ll,il,kl,P] 
                            rec = precision_recall[iou_ind,ll,il,kl,R] 
                            #recall must be at least target recall
                            if rec >= recall:
                                if prec > max_prec[ll]:
                                    max_prec[ll] = prec 
                    max_precisions[iou_ind,:,jl] = max_prec


            #calculate average precision, and record precision/recall
            for iou_ind in range(len(self.iou_thresholds)):
                for diff in range(max_difficulty+1):
                    avg_prec[iou_ind,diff,il] = max_precisions[iou_ind,diff,:].mean() 
                    all_max_precisions[iou_ind,diff,il] = max_precisions[iou_ind,diff,:]



        #calulate  map
        mean_ap = np.zeros(len(self.iou_thresholds))
        for iou_ind in range(len(self.iou_thresholds)):
            mean_ap[iou_ind] = avg_prec[iou_ind,0,np.where(avg_prec[iou_ind,0,:]>=0)].mean()

        #consolidate errors
        all_errors = {}
        all_errors['loc'] = loc_errors
        all_errors['nms'] = nms_errors
        all_errors['missed'] = missed_errors 
        all_errors['con'] = con_errors 
        all_errors['bg'] = bg_errors 
        all_errors['correct'] = correct_cnt

        #make image counts
        image_counts = {}
        image_counts['skipped_images'] = skipped_images
        image_counts['no_boxes_outputted'] = no_boxes_outputted
        image_counts['total images'] = len(all_image_names)

        #assign useful data structs as member vars for easy access later
        self.right_wrong = right_wrong
        self.gt_total = gt_total 
        self.precision_recall = precision_recall
        self.all_errors = all_errors
        self.scored_errors = scored_errors
        self.avg_prec = avg_prec
        self.all_max_precisions = all_max_precisions
        self.image_counts = image_counts
        self.mean_ap = mean_ap 

        self.gt_names = gt_names
        self.no_names = no_names

        #return [mean_ap, avg_prec, all_max_precisions, all_errors, gt_total, image_counts]
        return mean_ap[0]















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
             'Home_003_1',
             #'Home_002_1',
             #'Home_003_2',
             #'test',
             #'Office_001_1'
             ] 

    max_difficulty=4
    chosen_ids = range(6)
    dataset = GetDataSet.get_fasterRCNN_AVD(data_path,
                                            scene_list,
                                            chosen_ids=chosen_ids,
                                            fraction_of_no_box=0,
                                            max_difficulty=5)
    gt_boxes = dataset.get_original_bboxes()

    #get outputs from testing model
    trained_model_names = [
                           #'FRA_1-28_18',
                           #'FRA_1-5_19',
                           #'FRA_TD_1-5_archA_33',
                           #'FRA_TD_1-5_archA_5_39',
                           #'FRA_TD_1-5_archB_1_70',
                           #'FRA_TD_1-5_archB_2_50',
                           #'FRA_TD_1-5_archB1_0_60',
                           #'FRA_TD_1-28_archB1_0_100',
                           #'FRA_TD_1-5_archC_1_15',
                           #'FRA_TD_1-5_archD_0_68',

                           #'FRA_TD_1-28_archF_7_15',
                           # 'TDID_archA_0_7',
                            'TDID_GMU_archA_2_9_27.73549_0.80030',
                            'TDID_GMU_archA_2_6_30.02121_0.77436',
                           'TDID_GMU_archA_2_39_20.75395_0.86257',
                           #'FRA_TD_1-5_archF_2_6',
                           #'FRA_TD_1-5_archF_2_10',
                           #'FRA_TD_1-5_archF_2_15',
                           #'FRA_TD_1-5_archF_2_20',
                           #'FRA_TD_1-5_archF_2_25',
                           #'FRA_TD_1-5_archF_2_30',
                           #'FRA_TD_1-5_archF_2_35',
                           #'FRA_TD_1-5_archF_2_40',
                           #'FRA_TD_1-5_archF_2_45',
                           #'FRA_TD_1-5_archF_2_50',
                           #'FRA_TD_1-5_archF_2_55',
                           #'FRA_TD_1-5_archF_2_60',
                           
                           #'FRA_TD_1-28_archA2_0_0',
                           #'FRA_TD_1-28_archA2_0_2',
                           #'FRA_TD_1-28_archA2_0_4',
                           #'FRA_TD_1-28_archA2_0_7',
                           #'FRA_TD_1-28_archA2_0_9',
                           #'FRA_TD_1-28_archA2_0_10',
                           #'FRA_TD_1-28_archA2_0_12',
                           #'FRA_TD_1-28_archA2_0_14',
                           #'FRA_TD_1-28_archA2_0_16',
                           #'FRA_TD_1-28_archA2_0_18',
                           #'FRA_TD_1-28_archA2_0_20',
                           #'FRA_TD_1-5_archA2_0_86',
                          ]

    for model_name in trained_model_names:

        #with open('/playpen/ammirato/Data/Detections/' + 
        #          'FasterRCNN_AVD/recorded_detection_models/' + model_name+'.json') as f:
        with open('/playpen/ammirato/Data/Detections/' + 
                  'FasterRCNN_AVD/' + model_name+'.json') as f:
            det_results = json.load(f)

        evaluater = DetectorEvaluater(score_thresholds=np.linspace(0,1,111),
                                      recall_thresholds=np.linspace(0,1,11))
        #m_ap,ap,max_p,errors,gt_total, image_counts = evaluater.run(
        m_ap = evaluater.run(
                    det_results,gt_boxes,chosen_ids,
                    max_difficulty=max_difficulty,
                    difficulty_classifier=dataset.get_box_difficulty)
        #print 'MAPish {}'.format(ap[ap.nonzero()].mean())
        print 'MAPish {}     {}'.format(m_ap, model_name)
        print evaluater.image_counts

