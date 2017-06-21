import torch
import torch.utils.data
import torchvision.models as torch_models
import matplotlib
matplotlib.use('TkAgg')

import cv2

import pdb
#import ipdb
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import sys
import os
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import tkMessageBox
from tkFileDialog import askopenfilename

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
from PreDefinedSquareImageNet_model21 import PreDefinedSquareImageNet
import GetDataSet
import AlexNet
import SlidingWindowDetector as SWD

#from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn import network
from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.fast_rcnn.nms_wrapper import nms
from faster_rcnn.faster_rcnn_target_driven import FasterRCNN as FasterRCNN_targetDriven



class DetectionVisualizer(object):
    """
    Provides visualization of what a trained network has learned. 

    Assumes model in pytorch format
    """

    #TODO get rid of image trans
    def __init__(self,
                 dataloader, 
                 to_orig_img_trans, 
                 image_trans, 
                 display_trans=None, 
                 id_to_name_dict=None,
                 fasterRCNN_modelname=None,
                 targetDriven_modelname=None,
                 targets_path=None
                 ):
        
        self.dataloader = dataloader
        self.img_iter = iter(dataloader)
        self.to_orig_img_trans = to_orig_img_trans
        self.image_trans = image_trans
        self.display_trans = display_trans
        self.id_to_name_dict = id_to_name_dict
        self.fasterRCNN_modelname=fasterRCNN_modelname
        self.targetDriven_modelname = targetDriven_modelname
        self.targets_path=targets_path

        self.root = Tk.Tk() 
        self.root.wm_title("Detection Vis")

        self.main_fig = Figure(figsize=(5, 4), dpi=100)
        self.main_img_axis = self.main_fig.add_subplot(1,1,1)

        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(self.main_fig, master=self.root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        #canvas.mpl_connect('key_press_event', self.on_key_event)
        canvas.mpl_connect('button_press_event', self.on_click)

        #menubar things
        self.menubar = Tk.Menu(self.root)
        det_model_menu = Tk.Menu(self.menubar,tearoff=0)
        det_model_menu.add_command(label='FasterRCNN', 
                                   command=self.load_fasterRCNN_model)
        det_model_menu.add_command(label='TargetDriven', 
                                   command=self.load_targetDriven_model)
        self.menubar.add_cascade(label='Load Detection Model',
                                 menu=det_model_menu)
        self.root.config(menu=self.menubar)

        self.all_buttons_frame = Tk.Frame(self.root)
        self.all_buttons_frame.pack(side=Tk.BOTTOM)
      
        button = Tk.Button(master=self.all_buttons_frame, text='Next',
                    command=self.get_next_image)
        button.grid(row=1,column=1)
        button = Tk.Button(master=self.all_buttons_frame, text='Prev',
                    command=self.get_prev_image)
        button.grid(row=2,column=1)
        button = Tk.Button(master=self.all_buttons_frame, text='Run SWD',
                    command=self.run_sliding_window_detector)
        button.grid(row=3,column=1)
        button = Tk.Button(master=self.all_buttons_frame, text='Run FasterRCNN',
                    command=self.run_fasterRCNN_detector)
        button.grid(row=4,column=1)
        button = Tk.Button(master=self.all_buttons_frame, text='Run TargetDriven',
                    command=self.run_targetDriven_detector)
        button.grid(row=5,column=1)
        button = Tk.Button(master=self.all_buttons_frame, text='Quit',
                    command=self._quit)
        button.grid(row=6,column=1)


        nav_names = ['forward','backward','rotate_ccw','rotate_cw',
                     'left', 'right']
        counter = 1 
        for name in nav_names:
            button = Tk.Button(master=self.all_buttons_frame,
                               text=name, 
                               command=lambda n=name: 
                                            self.move(n))
            button.grid(row=counter,column=2)
            counter += 1

        target_names = os.listdir(targets_path)
        target_names.sort()

        self.targetVar = Tk.StringVar(self.all_buttons_frame)
        self.targetVar.set('Pick a target')
        self.target_menu = Tk.OptionMenu(self.all_buttons_frame,self.targetVar,
                                   *target_names)
        self.targetVar.trace('w', self.target_menu_selected)
        self.target_menu.grid(row=2, column=3)

        Tk.Label(self.all_buttons_frame,text='Choose Target Instance'
                                            ).grid(row=1,
                                                  column=3)


        self.main_canvas = canvas
        self.cur_img_index = -1 
        self.main_img_boxtl = None 
        self.rect = None
        self.get_next_image()


    def _quit(self):
        self.root.quit()
        self.root.destroy()


    def get_and_run_next_image(self):
        """
        Gets the next image, and runs it through the network.

        """ 
        #get the next image and label, store in class
        #img,label = self.img_iter.next()
        self.cur_img_index += 1
        img,label = self.dataloader.dataset[self.cur_img_index]
        img = img.unsqueeze(0)
        self.main_img_tensor = img
        self.main_label = label
        self.main_img_np = self.to_orig_img_trans(img).astype(np.uint8)
       
        #run the image through the model 
        pred_class, pred_score = self.run_image_through_model(img)

        #get the display image
        self.make_display_image()

        #display the image and prediction
        self.main_img_axis.imshow(self.display_img)
        self.main_img_axis.set_title( 'GT: {} Pred: {} Score: {:.2f}'.format(
                                                            self.main_label,
                                                            pred_class,
                                                            pred_score))
        #clear the occ img
        self.occ_img = None
        self.occ_img_axis.imshow(np.zeros((5,5)), vmin=0, vmax=1)
        self.occ_img_axis.set_title('')

        #display everything
        self.main_canvas.show()        

    def get_and_run_prev_image(self):
        #move back two, so the next image after that is back one
        self.cur_img_index -= 2 
        self.get_and_run_next_image()
        

    def run_image_through_model(self,img):
        #prediction = self.model(torch.autograd.Variable(img))
        prediction,self.model_outputs = self.model.forward_vis(
                                                torch.autograd.Variable(img))
        #prediction= self.model_outputs[-1]
        pred_score, pred_class = torch.nn.functional.softmax(prediction).topk(1)
        pred_score = pred_score.data[0][0]
        pred_class = pred_class.data[0][0]
        return [pred_class, pred_score]


    def label_menu_selected(self, *args):
        label = int(self.label_menu_value.get())
        correct_inds,wrong_inds = self.right_wrong_indices[label]

        #populate correct menu options with image indexes
        self.correct_menu['menu'].delete(0,'end')
        self.correct_menu_value.set(str(correct_inds[0]))
        print (correct_inds[-1])
        for ind in correct_inds:
            self.correct_menu['menu'].add_command(label=str(ind),
                              command=Tk._setit(self.correct_menu_value,ind))
        #do same for wrong menu 
        self.wrong_menu['menu'].delete(0,'end')
        self.wrong_menu_value.set(str(wrong_inds[0]))
        for ind in wrong_inds:
            self.wrong_menu['menu'].add_command(label=str(ind),
                              command=Tk._setit(self.wrong_menu_value,ind))


    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate



    def move(self,movement):
        """
        Moves to the image that is in the given direciton.
        """
        new_image_name = self.main_label[-1][movement]
        if not(new_image_name == ''):
            self.cur_img_index = self.dataloader.dataset.get_name_index(
                                                            new_image_name)

        #move back one, then get the next one
        self.cur_img_index -= 1
        self.get_next_image()


    def get_next_image(self):
        """
        Gets the next image.

        """ 
        self.main_img_axis.clear()
        self.cur_img_index += 1
        img,label = self.dataloader.dataset[self.cur_img_index]
        img = img.unsqueeze(0)
        self.main_img_tensor = img
        self.main_label = label
        self.main_img_np = self.to_orig_img_trans(img).astype(np.uint8)
       
        #run the image through the model 
        #pred_class, pred_score = self.run_image_through_model(img)

        #get the display image
        if self.display_trans is not None:
            self.display_img = self.display_trans(self.main_img_np.copy())
        else:
            self.display_img = self.main_img_np.copy()

        #remove previous box if it exists
        #if not(self.rect is None):
        #    self.rect.remove()
        self.rect = None

        #display the image and prediction
        self.main_img_axis.imshow(self.display_img)
        
        #display everything
        self.main_canvas.show()        

    def get_prev_image(self):
        self.cur_img_index -= 2
        self.get_next_image()

    def on_click(self,event):
        #user clicked mouse      
 
        #for drawing occluding boxes on main img 
        if event.inaxes == self.main_img_axis:
            if self.main_img_boxtl is None:
                self.main_img_boxtl = [int(event.xdata),int(event.ydata)]
            else:
                #define occluding box
                box = [
                        self.main_img_boxtl[0],  
                        self.main_img_boxtl[1],  
                        int(event.xdata),
                        int(event.ydata)  
                      ]
                #clear for next box
                self.main_img_boxtl = None
                #if the box is not valid don't draw it 
                if(box[0]>box[2] or box[1]>box[3]):
                    return

                crop_img = self.main_img_np[box[1]:box[3],box[0]:box[2],:]

                #apply the neccessary image transforms to prepare the 
                #image for the model(e.g. to pytorch Tensor)
                ten_img = self.image_trans(crop_img)

                #run the image through the network
                pred_class, pred_score = self.run_image_through_model(
                                                        ten_img.unsqueeze(0))
                 
                self.main_img_axis.set_title( 'Pred: {} Score: {:.2f}'
                                                    .format(
                                                            pred_class,
                                                            pred_score))



                #remove previous box if it exists
                if not(self.rect is None):
                    self.rect.remove()

                #draw everything
                self.main_img_axis.imshow(self.display_img) 
                self.rect = patches.Rectangle((box[0],box[1]),
                                         box[2]-box[0],
                                         box[3]-box[1],
                                         linewidth=2,
                                         edgecolor='r',
                                         facecolor='none')
                self.main_img_axis.add_patch(self.rect)
                self.main_canvas.show()



    def run_sliding_window_detector(self):

        tkMessageBox.showinfo('message', 'Not implemented yet!')
        return
        ##run the detector
        #detector = SWD.SlidingWindowDetector(self.model,
        #                       image_trans=self.dataloader.dataset.transform)
        #results = detector(self.main_img_tensor.squeeze())

        ##display the results
        #for box in results:
        #    rect = patches.Rectangle((box[0],box[1]),
        #                             box[2]-box[0],
        #                             box[3]-box[1],
        #                             linewidth=2,
        #                             edgecolor='r',
        #                             facecolor='none',
        #                             alpha=.5)
        #    self.main_img_axis.add_patch(rect)
        #   
        #    #write class and score
        #    self.main_img_axis.text(box[0]+15, box[1]+15,str(box[4]), 
        #                                fontsize=5) 
        #    self.main_img_axis.text(box[0]+15, box[1]+30,str(box[5]),
        #                                fontsize=5) 
        #    self.main_canvas.show()
       

 
    def run_fasterRCNN_detector(self):


        if not hasattr(self,'fasterRCNN_model'):
            tkMessageBox.showinfo('message', 'Model not loaded yet!')
            return
            
        max_per_image = 300 
        thresh = 0.05

        im_data = self.main_img_tensor
        im_data=im_data.numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        im_info = np.zeros((1,3))
        im_info[0,:] = [im_data.shape[1],im_data.shape[2],1]


        cls_prob, bbox_pred, rois = self.fasterRCNN_model(im_data, im_info)
        scores = cls_prob.data.cpu().numpy()
        boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    #if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        #pred_boxes = clip_boxes(pred_boxes, image.shape)
        pred_boxes = clip_boxes(pred_boxes, im_data.shape[1:])
        boxes = pred_boxes
    #else:
    #    # Simply repeat the boxes, once for each class
    #    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        all_boxes = [[] for _ in xrange(self.dataloader.dataset.get_num_classes())]
        #separate boxes by class, non maximum supression
        # skip j = 0, because it's the background class
        for j in xrange(1, self.dataloader.dataset.get_num_classes()):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .3)
            cls_dets = cls_dets[keep, :]
            all_boxes[j] = cls_dets
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:,-1]
                                      for j in xrange(1, dataloader.dataset.get_num_classes())])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, dataloader.dataset.get_num_classes()):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        self.draw_boxes_on_image(all_boxes)


    def target_menu_selected(self, *args):
        print self.targetVar.get()

    def run_targetDriven_detector(self):

        
        if not hasattr(self,'targetDriven_model'):
            tkMessageBox.showinfo('message', 'Model not loaded yet!')
            return
           
        if self.targetVar.get() == 'Pick a target':
            tkMessageBox.showinfo('message', 'Pick a target image!')
            return
            
        target_data = cv2.imread(os.path.join(self.targets_path,self.targetVar.get()))
        target_data = np.expand_dims(target_data,axis=0)
 
        max_per_image = 300 
        thresh = 0.05

        im_data = self.main_img_tensor
        im_data=im_data.numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        im_info = np.zeros((1,3))
        im_info[0,:] = [im_data.shape[1],im_data.shape[2],1]


        cls_prob, bbox_pred, rois = self.targetDriven_model(target_data,im_data, im_info)
        scores = cls_prob.data.cpu().numpy()
        boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    #if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        #pred_boxes = clip_boxes(pred_boxes, image.shape)
        pred_boxes = clip_boxes(pred_boxes, im_data.shape[1:])
        boxes = pred_boxes
    #else:
    #    # Simply repeat the boxes, once for each class
    #    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        all_boxes = [[] for _ in xrange(0,2)]
        #separate boxes by class, non maximum supression
        # skip j = 0, because it's the background class
        for j in xrange(1,2):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, .3)
            cls_dets = cls_dets[keep, :]
            all_boxes[j] = cls_dets
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:,-1]
                                      for j in xrange(1,2)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, 2):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        self.draw_boxes_on_image(all_boxes)


        #tkMessageBox.showinfo('message', 'Not implemented yet!')
        return

    def load_fasterRCNN_model(self):
        ''' load a trained Faster RCNN model '''
        #get the filename
        model_name = askopenfilename()
        #create the empty network, and load in the saved parameters
        net = FasterRCNN(classes=dataset.get_class_names(), debug=False)
        network.load_net(model_name, net)
        net.cuda()
        net.eval()
        self.fasterRCNN_model = net         
        tkMessageBox.showinfo('message', 'Done loading!!')


    def load_targetDriven_model(self):
        ''' load a trained Faster RCNN model '''
        #get the filename
        model_name = askopenfilename()
        #create the empty network, and load in the saved parameters
        net = FasterRCNN_targetDriven(classes=dataset.get_class_names(), debug=False)
        network.load_net(model_name, net)
        net.cuda()
        net.eval()
        self.targetDriven_model = net         
        tkMessageBox.showinfo('message', 'Done loading!!')


    def draw_boxes_on_image(self,boxes_by_class):
        display_thresh = .3
        for cid,class_boxes in enumerate(boxes_by_class):
            for box in class_boxes:
                if box[4] < display_thresh:
                    continue
                rect = patches.Rectangle((box[0],box[1]),
                                         box[2]-box[0],
                                         box[3]-box[1],
                                         linewidth=2,
                                         edgecolor='r',
                                         facecolor='none',
                                         alpha=.5)
                self.main_img_axis.add_patch(rect)
               
                #write class and score
                self.main_img_axis.text(box[0]+15,
                                        box[1]+15,
                                        str(cid), 
                                        fontsize=15,
                                        bbox=dict(facecolor='red')) 
                self.main_img_axis.text(box[0]+15,
                                        box[1]+30,
                                        str(box[4]), 
                                        fontsize=15,
                                        bbox=dict(facecolor='red')) 
                self.main_canvas.show()
       





##################################################################
##################################################################
##################################################################


#USER INPUT
data_path = '/playpen/ammirato/Data/HalvedRohitData/'
scene_list=[
            # 'Home_006_1',
             'Home_008_1',
            # 'Home_002_1'
             ]


dataset = GetDataSet.get_fasterRCNN_AVD(data_path,
                                        scene_list,
                                        preload=False,
                                        chosen_ids=range(0,28))#[0,1,2,3,4,5])
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         collate_fn=AVD.collate)

to_orig_img_trans = GetDataSet.get_fasterRCNN_AVD_to_orig_image_trans()
bgr_rgb_trans = AVD_transforms.BGRToRGB()

targets_path = '/playpen/ammirato/Data/big_bird_crops_160/'

#create GUI
window =  DetectionVisualizer(dataloader,to_orig_img_trans, 
                              dataset.transform, 
                              display_trans=bgr_rgb_trans,
                              targets_path=targets_path)



