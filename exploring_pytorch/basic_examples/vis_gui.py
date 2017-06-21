import torch
import torch.utils.data
import torchvision.models as torch_models
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
from PreDefinedSquareImageNet_model21 import PreDefinedSquareImageNet
import GetDataSet
import AlexNet
import SlidingWindowDetector as SWD


#TODO - make one 'ImageLoader' class that detection gui and main gui inheret




class NeuralNetworkVisualizer(object):
    """
    Provides visualization of what a trained network has learned. 

    Assumes model in pytorch format
    """

    #TODO get rid of image trans
    def __init__(self,dataloader, model,to_orig_img_trans, image_trans, 
                 display_trans=None, id_to_name_dict=None,
                 det_dataloader=None):
        
        self.dataloader = dataloader
        self.det_dataloader = det_dataloader
        self.img_iter = iter(dataloader)
        self.model = model
        self.to_orig_img_trans = to_orig_img_trans
        self.image_trans = image_trans
        self.display_trans = display_trans
        self.id_to_name_dict = id_to_name_dict
        

        #some placeholders for later
        self.main_img_boxtl = None
        self.main_img_boxbr = None
        self.occ_img = None 
        self.cur_img_index = -1 

        self.root = Tk.Tk()
        self.root.wm_title("NN Vis")

        self.main_fig = Figure(figsize=(5, 4), dpi=100)
        self.main_img_axis = self.main_fig.add_subplot(1,2,1)
        self.occ_img_axis = self.main_fig.add_subplot(1,2,2)

        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(self.main_fig, master=self.root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        #toolbar = NavigationToolbar2TkAgg(canvas, self.root)
        #toolbar.update()
        #canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        canvas.mpl_connect('key_press_event', self.on_key_event)
        canvas.mpl_connect('button_press_event', self.on_click)

        self.main_canvas = canvas


        #set up menu options
        self.menubar = Tk.Menu(self.root)
        self.menubar.add_command(label='Run Full Test', 
                                 command=self.run_full_test)
        self.menubar.add_command(label='Open Detection GUI', 
                                 command=self.open_detection_gui)
        self.root.config(menu=self.menubar)

        #set up frames to hold buttons
        self.all_buttons_frame = Tk.Frame(self.root)
        self.all_buttons_frame.pack(side=Tk.BOTTOM)
        self.layer_button_frame = Tk.Frame(self.all_buttons_frame)
        self.layer_button_frame.pack(side=Tk.LEFT)
        self.nav_button_frame = Tk.Frame(self.all_buttons_frame)
        self.nav_button_frame.pack(side=Tk.RIGHT)
      
        button = Tk.Button(master=self.nav_button_frame, text='Quit',
                           command=self._quit)
        #button.pack(side=Tk.BOTTOM)
        button.grid(row=4,column=1)
        button = Tk.Button(master=self.nav_button_frame, text='Next',
                    command=self.get_and_run_next_image)
        #button.pack(side=Tk.TOP)
        button.grid(row=1,column=1)
        button = Tk.Button(master=self.nav_button_frame, text='Prev',
                    command=self.get_and_run_prev_image)
        #button.pack(side=Tk.TOP)
        button.grid(row=2,column=1)
        button = Tk.Button(master=self.nav_button_frame,
                           text='Show Occlusion Image',
                           command=self.create_occlusion_performance_img)
        #button.pack(side=Tk.TOP)
        button.grid(row=3,column=1)


        self.label_menu_value = Tk.StringVar(self.nav_button_frame)
        self.label_menu_value.set('Run full test first')
        self.label_menu = Tk.OptionMenu(self.nav_button_frame,self.label_menu_value,
                                   self.label_menu_value.get())
        self.label_menu_value.trace('w', self.label_menu_selected)
        #label_menu.pack(side=Tk.BOTTOM)
        self.label_menu.grid(row=2, column=3)

        Tk.Label(self.nav_button_frame,text='Choose label to view').grid(row=1,
                                                                      column=3)

        self.correct_menu_value = Tk.StringVar(self.nav_button_frame)
        self.correct_menu_value.set('Choose Label First')
        self.correct_menu = Tk.OptionMenu(self.nav_button_frame,self.correct_menu_value,
                                   self.correct_menu_value.get())
        self.correct_menu_value.trace('w', self.correct_menu_selected)
        #label_menu.pack(side=Tk.BOTTOM)
        self.correct_menu.grid(row=4, column=3)

        Tk.Label(self.nav_button_frame,text='Choose Correctly' + 
                                            'Classified Image(by index)'
                                            ).grid(row=3,
                                                  column=3)

        self.wrong_menu_value = Tk.StringVar(self.nav_button_frame)
        self.wrong_menu_value.set('Choose Label First')
        self.wrong_menu = Tk.OptionMenu(self.nav_button_frame,self.wrong_menu_value,
                                   self.wrong_menu_value.get())
        self.wrong_menu_value.trace('w', self.wrong_menu_selected)
        self.wrong_menu.grid(row=6, column=3)

        Tk.Label(self.nav_button_frame,text='Choose Wrongly' + 
                                            'Classified Image(by index)'
                                            ).grid(row=5,
                                                  column=3)





        #set up buttons for visualizing each layer
        #get an image from the dataset and run it through the model
        temp_iter = iter(dataloader)
        img,label = temp_iter.next() 
        _,layer_outputs = self.model.forward_vis(torch.autograd.Variable(img))
        layer_buttons = []
        counter = 0 
        for layer in layer_outputs:
            name = ('Layer ' + str(counter))
            button = Tk.Button(master=self.layer_button_frame,
                               text=name, 
                               command=lambda n=name: 
                                            self.vis_layer_activations(n))
            button.pack(side=Tk.BOTTOM)
            row = int(counter/4)
            col = counter %4
            
#            button.grid(column=col, row=row)
 
            counter +=1




        self.get_and_run_next_image()

        Tk.mainloop()
        # If you put root.destroy() here, it will cause an error if
        # the window is closed with the window manager.



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


    def create_occlusion_performance_img(self, box_size=None):
        """
        Creates image showing performance as a function of occlusion.
    
        Assumes square input image
        """

        if self.occ_img is None:
            #get the original image and the true label
            img = self.main_img_np
            label = self.main_label

            ##create data structures for tracking where occlusion is
            #build the occluding box(a bunch of zeros)
            num = 10  
            img_size = img.shape[0] 
            if box_size is None:
                box_size = int(img_size/num)
            box = np.zeros((box_size,box_size,3))

            #get the rows/cols to place the center of the box as it is 
            #moved around the image
            start = (box_size/2) 
            end = img_size - (box_size/2) - 1 
            rows = np.linspace(start,end,num).astype(np.int)
            cols = np.linspace(start,end,num).astype(np.int)
        
            #hold the prediction results for each box location
            results = np.zeros((len(rows),len(cols)))
            
            #for each box location, ...
            for il in range(len(rows)):
                for jl in range(len(cols)):
                    row = rows[il]
                    col = cols[jl]

                    box = [(col-box_size/2),(row-box_size/2), 
                           (col+box_size/2),(row+box_size/2),
                           3]
                    pred_class, pred_score = self.run_occluded_image(box)

                    #record prediction result
                    if pred_class == label:
                        results[il,jl] = pred_score

            self.occ_img = results

        self.occ_img_axis.imshow(self.occ_img, vmin=0, vmax=1)
        self.occ_img_axis.set_title('Min: {:.2f} Max: {:.2f}'.format(
                                                          self.occ_img.min(),
                                                          self.occ_img.max()))
        #display everything
        self.main_canvas.show()        


    def make_display_image(self):
        if self.display_trans is not None:
            self.display_img = self.display_trans(self.main_img_np.copy())
        else:
            self.display_img = self.main_img_np.copy()

    def on_key_event(self,event):
        print('you pressed %s' % event.key)
        #key_press_handler(event, canvas, toolbar)

    def run_occluded_image(self,box):
        #shorten var name
        img = self.main_img_np

        #save the region to be occluded
        orig_region = img[box[1]:box[3],box[0]:box[2],:].copy() 
        #put in the occlusion
        img[box[1]:box[3],box[0]:box[2],:] = np.zeros((box[3]-box[1],
                                                       box[2]-box[0],
                                                       3))
        #apply the neccessary image transforms to prepare the 
        #image for the model(e.g. to pytorch Tensor)
        #TODO - work with transformed image to speed up?
        ten_img = self.image_trans(img)

        #run the image through the network
        pred_class, pred_score = self.run_image_through_model(
                                                ten_img.unsqueeze(0))
        #restore the occluded region
        img[box[1]:box[3],box[0]:box[2],:] = orig_region 

        return [pred_class,pred_score]

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

                #reset display image
                self.make_display_image()
                self.display_img[box[1]:box[3],box[0]:box[2],:] = np.zeros(
                                            (box[3]-box[1],box[2]-box[0],3))

                pred_class, pred_score = self.run_occluded_image(box)
                 
                self.main_img_axis.set_title( 'GT: {} Pred: {} Score: {:.2f}'
                                                    .format(
                                                            self.main_label,
                                                            pred_class,
                                                            pred_score))




                #draw everything
                self.main_img_axis.imshow(self.display_img) 
                self.main_canvas.show()


    def vis_layer_activations(self,button_name):
        index = int(str.split(button_name)[1])

        #get feature map and convert from Tensor to numpy
        feature_maps = self.model_outputs[index]
        #remove batch demension
        feature_maps = feature_maps.data.squeeze(0)

        #make sure tensor is on cpu before converting to numpy
        npimg = feature_maps.cpu().numpy()
       

        act_fig = plt.figure(0)
 
        if len(npimg.shape) == 3: #conv feature maps
            #make HxWxC
            npimg = np.transpose(npimg,(1,2,0))



            num_rows = int(np.ceil(np.sqrt(npimg.shape[2])))
            num_cols = num_rows
            grid = gridspec.GridSpec(
                                        num_rows,
                                        num_cols,
                                        )
            #for colormap
            vmin = npimg.min()
            vmax = npimg.max()
            scale = 255.0/vmax

            for jl in range(npimg.shape[2]):
                img = npimg[:,:,jl]

#                img = (img + vmin) *scale

                ax = plt.Subplot(act_fig,grid[jl]) 
                #ax.imshow(img.astype(np.uint8), vmin=vmin, vmax=vmax)
                ax.imshow(img, vmin=vmin, vmax=vmax)
                act_fig.add_subplot(ax)
                ax.set_xticks([])
                ax.set_yticks([])

           # ax = plt.Subplot(act_fig,inner_grid[jl+1])
           # txt = ax.text(.1,.5,'Min: {}\n Max: {}'.format(vmin, vmax))
           # txt.set_clip_on(False)
           # act_fig.add_subplot(ax)
        
            plt.draw()
            plt.pause(.001)


    def run_full_test(self):

        acc_by_class = {} 
        right_wrong_indices = {}
        dataset = self.dataloader.dataset
        num_correct = 0
        total = 0
        for il in range(len(dataset)):
            #display progress
            if il % 200 == 0:
                print('{}/{}'.format(il,len(dataset)))        
   
            #get the images and labels for this batch 
            img,label = dataset[il]

            #if this is the first example from this class
            if label not in acc_by_class.keys():
                acc_by_class[label] = [0,0,0]#correct,total,accuracy
                right_wrong_indices[label] = [[],[]] 
 
            #forward pass,  get 
            pred = self.model.forward(torch.autograd.Variable(img.unsqueeze(0)))
            pred_score, pred_class = torch.nn.functional.softmax(pred).topk(1)
    
            #if predicition is correct, record 
            if label == pred_class.data.numpy()[0,0]:
                num_correct += 1
                acc_by_class[label][0] +=1
                right_wrong_indices[label][0].append(il)
            else:
                right_wrong_indices[label][1].append(il)

            #add to total 
            total += 1
            acc_by_class[label][1] +=1

        #calculate accuracies
        for label in acc_by_class.keys():
            acc_by_class[label][2] = 100*(acc_by_class[label][0]/
                                          float(acc_by_class[label][1]))

        plt.figure()
        plt.bar(range(len(acc_by_class)),[x for _,_,x in acc_by_class.values()])
        plt.xticks(range(len(acc_by_class)),acc_by_class.keys())
        plt.title('Histogram of Accuracies by Class')
        plt.xlabel('class id')
        plt.ylabel('Accuracy')
        plt.draw()
        plt.pause(.001)

        self.acc_by_class = acc_by_class 
        self.right_wrong_indices = right_wrong_indices 


        #populate label choices
        label_ids = acc_by_class.keys()
        label_menu_options = [] 
        if self.id_to_name_dict is not None:        
            for nid in label_ids:
                name = self.id_to_name_dict[nid]
                label_menu_options.append(name)
        else:
            for nid in label_ids:
                label_menu_options.append(str(nid))

        #remove old choices
        self.label_menu['menu'].delete(0,'end')
        self.label_menu_value.set(label_menu_options[0])  

        for choice in label_menu_options:
           self.label_menu['menu'].add_command(label=choice,
                              command=Tk._setit(self.label_menu_value,choice))

        #set first option
        breakp = 1

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

    def correct_menu_selected(self, *args):
        print(self.correct_menu_value.get()) 
        #move back one, so the next image is the one we want 
        self.cur_img_index = int(self.correct_menu_value.get()) - 1 
        self.get_and_run_next_image()
    
    def wrong_menu_selected(self, *args):
        #move back one, so the next image is the one we want 
        self.cur_img_index = int(self.wrong_menu_value.get()) - 1 
        self.get_and_run_next_image()

    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate


    def open_detection_gui(self):
        if self.det_dataloader is None:
            print 'No Detection'
            return        

        self.det_root = Tk.Toplevel(self.root) 
        self.det_root.wm_title("Detection Vis")

        self.det_main_fig = Figure(figsize=(5, 4), dpi=100)
        self.det_main_img_axis = self.det_main_fig.add_subplot(1,1,1)

        # a tk.DrawingArea
        det_canvas = FigureCanvasTkAgg(self.det_main_fig, master=self.det_root)
        det_canvas.show()
        det_canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        #canvas.mpl_connect('key_press_event', self.on_key_event)
        det_canvas.mpl_connect('button_press_event', self.det_on_click)



        self.det_all_buttons_frame = Tk.Frame(self.det_root)
        self.det_all_buttons_frame.pack(side=Tk.BOTTOM)
      
        button = Tk.Button(master=self.det_all_buttons_frame, text='Next',
                    command=self.get_next_det_image)
        button.grid(row=1,column=1)
        button = Tk.Button(master=self.det_all_buttons_frame, text='Prev',
                    command=self.get_prev_det_image)
        button.grid(row=2,column=1)
        button = Tk.Button(master=self.det_all_buttons_frame, text='Run SWD',
                    command=self.run_sliding_window_detector)
        button.grid(row=3,column=1)
        button = Tk.Button(master=self.det_all_buttons_frame, text='Run FasterRCNN',
                    command=self.run_fasterRCNN_detector)
        button.grid(row=4,column=1)


        nav_names = ['forward','backward','rotate_ccw','rotate_cw',
                     'left', 'right']
        counter = 1 
        for name in nav_names:
            button = Tk.Button(master=self.det_all_buttons_frame,
                               text=name, 
                               command=lambda n=name: 
                                            self.move(n))
            button.grid(row=counter,column=2)
            counter += 1

        self.det_main_canvas = det_canvas
        self.det_cur_img_index = -1 
        self.det_main_img_boxtl = None 
        self.det_rect = None
        self.get_next_det_image()

    def move(self,movement):
        """
        Moves to the image that is in the given direciton.
        """
        new_image_name = self.det_main_label[-1][movement]
        if not(new_image_name == ''):
            self.det_cur_img_index = self.det_dataloader.dataset.get_name_index(
                                                            new_image_name)

        #move back one, then get the next one
        self.det_cur_img_index -= 1
        self.get_next_det_image()


    def get_next_det_image(self):
        """
        Gets the next image.

        """ 
        self.det_main_img_axis.clear()
        self.det_cur_img_index += 1
        img,label = self.det_dataloader.dataset[self.det_cur_img_index]
        img = img.unsqueeze(0)
        self.det_main_img_tensor = img
        self.det_main_label = label
        self.det_main_img_np = self.to_orig_img_trans(img).astype(np.uint8)
       
        #run the image through the model 
        #pred_class, pred_score = self.run_image_through_model(img)

        #get the display image
        if self.display_trans is not None:
            self.det_display_img = self.display_trans(self.det_main_img_np.copy())
        else:
            self.det_display_img = self.det_main_img_np.copy()

        #remove previous box if it exists
        #if not(self.det_rect is None):
        #    self.det_rect.remove()
        self.det_rect = None

        #display the image and prediction
        self.det_main_img_axis.imshow(self.det_display_img)
        
        #display everything
        self.det_main_canvas.show()        

    def get_prev_det_image(self):
        self.det_cur_img_index -= 2
        self.get_next_det_image()

    def det_on_click(self,event):
        #user clicked mouse      
 
        #for drawing occluding boxes on main img 
        if event.inaxes == self.det_main_img_axis:
            if self.det_main_img_boxtl is None:
                self.det_main_img_boxtl = [int(event.xdata),int(event.ydata)]
            else:
                #define occluding box
                box = [
                        self.det_main_img_boxtl[0],  
                        self.det_main_img_boxtl[1],  
                        int(event.xdata),
                        int(event.ydata)  
                      ]
                #clear for next box
                self.det_main_img_boxtl = None
                #if the box is not valid don't draw it 
                if(box[0]>box[2] or box[1]>box[3]):
                    return

                crop_img = self.det_main_img_np[box[1]:box[3],box[0]:box[2],:]

                #apply the neccessary image transforms to prepare the 
                #image for the model(e.g. to pytorch Tensor)
                ten_img = self.image_trans(crop_img)

                #run the image through the network
                pred_class, pred_score = self.run_image_through_model(
                                                        ten_img.unsqueeze(0))
                 
                self.det_main_img_axis.set_title( 'Pred: {} Score: {:.2f}'
                                                    .format(
                                                            pred_class,
                                                            pred_score))



                #remove previous box if it exists
                if not(self.det_rect is None):
                    self.det_rect.remove()

                #draw everything
                self.det_main_img_axis.imshow(self.det_display_img) 
                self.det_rect = patches.Rectangle((box[0],box[1]),
                                         box[2]-box[0],
                                         box[3]-box[1],
                                         linewidth=2,
                                         edgecolor='r',
                                         facecolor='none')
                self.det_main_img_axis.add_patch(self.det_rect)
                self.det_main_canvas.show()



    def run_sliding_window_detector(self):
        #run the detector
        detector = SWD.SlidingWindowDetector(self.model,
                               image_trans=self.dataloader.dataset.transform)
        results = detector(self.det_main_img_tensor.squeeze())

        #display the results
        for box in results:
            rect = patches.Rectangle((box[0],box[1]),
                                     box[2]-box[0],
                                     box[3]-box[1],
                                     linewidth=2,
                                     edgecolor='r',
                                     facecolor='none',
                                     alpha=.5)
            self.det_main_img_axis.add_patch(rect)
           
            #write class and score
            self.det_main_img_axis.text(box[0]+15, box[1]+15,str(box[4]), 
                                        fontsize=5) 
            self.det_main_img_axis.text(box[0]+15, box[1]+30,str(box[5]),
                                        fontsize=5) 
            self.det_main_canvas.show()
        
    def run_fasterRCNN_detector(self):

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
load_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' + 
             'exploring_pytorch/saved_models/recorded_models/')
model_name = 'model_38_2_0.903919560562.p'



dataset = GetDataSet.get_alexnet_AVD(data_path,scene_list)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         collate_fn=AVD.collate)

det_dataset = GetDataSet.get_alexnet_AVD(data_path,scene_list,detection=True)
det_dataloader = torch.utils.data.DataLoader(det_dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         collate_fn=AVD.collate)

#convert from tensor image to original RGB image
denorm_trans = AVD_transforms.NormalizeRange(0.0,255.0,
                                             from_min=-2.16518,
                                             from_max=2.65179)
to_numpy_trans = AVD_transforms.ToNumpy()
bgr_rgb_trans = AVD_transforms.BGRToRGB()
to_orig_img_trans = AVD_transforms.Compose([
                                            denorm_trans,
                                            to_numpy_trans,
                                      #      bgr_rgb_trans,
                                            ])


#load alexnet model 
#model = torch_models.AlexNet(28)
model = AlexNet.AlexNet(28)
model.load_state_dict(torch.load(load_path+model_name))



#create GUI
window =  NeuralNetworkVisualizer(dataloader,model,to_orig_img_trans, 
                                  dataset.transform, 
                                  display_trans=bgr_rgb_trans,
                                  det_dataloader=det_dataloader)



