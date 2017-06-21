import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
from PreDefinedSquareImageNet_model10 import PreDefinedSquareImageNet
import time
import numpy as np
#import matplotlib.pyplot as plt


class SlidingWindowDetector(object):
    """
    Makes a sliding window detector.

    Takes in an object classifier, and uses it to classify a cropped portion
    of the input image. 

    Uses pytorch.
    """


    def __init__(self, classifier, image_trans=None, cuda=False):
        self.classifier = classifier
        self.image_trans = image_trans
        self.cuda = cuda

        if self.cuda:
            self.classifier.cuda()

    def __call__(self, img):
        """
        Outputs boxes for detected boxes in the given image

        Assumes tensor image of CxHxW

        """
        #TODO: allow batches of images
        if len(img.size()) == 4:
            if(img.size()[0] != 1):#must be single image
                return -1
            img = img.squeeze()

        img = img.cpu().numpy()
        img = np.transpose(img,(1,2,0))


        #crop non overlapping square boxes of size max_box_dim
        max_box_dim = 100 
        num_rows = img.shape[0]/max_box_dim
        num_cols = img.shape[1]/max_box_dim

        #pick the top left corner location of each crop
        rows = np.linspace(0,img.shape[0]-max_box_dim,num_rows).astype(np.int)
        cols = np.linspace(0,img.shape[1]-max_box_dim,num_cols).astype(np.int)

        #store classification results(class,score) for each box
        results = np.zeros((len(rows)*len(cols),2))
        out_boxes = []

        #for each box location, ...
        for il in range(len(rows)):
            for jl in range(len(cols)):
                row = rows[il]
                col = cols[jl]

                cropped_img = img[row:row+max_box_dim,
                                  col:col+max_box_dim,
                                  :]
                #plt.imshow(cropped_img)

                if self.image_trans is not None:
                    cropped_img = self.image_trans(cropped_img)

                if self.cuda:
                    cropped_img.cuda()

                prediction = self.classifier(Variable(cropped_img.unsqueeze(0)))
                pred_score, pred_class = torch.nn.functional.softmax(
                                                    prediction).topk(1)

                #plt.title('Class: {}  Score: {}'.format(pred_class.data.numpy(),
                #                                        pred_score.data.numpy()))
                #plt.draw()
                #plt.pause(.001)
                #raw_input('Enter')
                results[il*len(cols) + jl,0] = pred_class.data.numpy()
                results[il*len(cols) + jl,1] = pred_score.data.numpy()

                out_boxes.append([col,row,col+max_box_dim,row+max_box_dim,
                                  pred_class.data.numpy()[0][0],
                                  pred_score.data.numpy()[0][0]])

        return out_boxes 
