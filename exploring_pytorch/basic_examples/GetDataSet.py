import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data

import numpy as np
import os

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms





def get_class_id_to_name_dict(root):
    map_file = open(os.path.join(root,'instance_id_map.txt'),'r')

    id_to_name_dict = {}

    for line in map_file:
        line = str.split(line)
        id_to_name_dict[int(line[1])] = line[0]


    return id_to_name_dict


    
#TODO get rid of org_img_dims

def get_standard_AVD(root, scene_list,
                     classification=False,
                     by_box=False, 
                     preload=False,
                     chosen_ids=range(28),
                     image_size=[128,128,3],
                     org_img_dims=[1920/2,1080/2,3]):
    """
    Returns a torch dataloader for the AVD dataset.

    Applies some standard transforms
    dataloader = get_standard_AVD('/path/to/data', ['scene1','scene2,...])


    ARGS:
        root: path to data. Parent of all scene directories
        scene_list: scenes to include
        
    KEYWORD ARGS:
        classification(False): if the data should be for classification task
                        (images cropped around bounding boxes)
        by_box(False): if data should be returned one box at a time
        preload(False): if images should all be loaded at initialization
        chosen_ids(range(28)): what instance ids to use
        image_size: all images resized to this
                    (after transforms/classification cropping)
        org_img_dims: original dimensions of the images
    """


    ##initialize transforms for the labels
    # - add background bounding boxes to labels
    back_trans = AVD_transforms.AddBackgroundBoxes(
                                num_to_add=1,
                                box_dimensions_range=[50,50,300,300],
                                image_dimensions=org_img_dims)
    #only consider boxes from the chosen classes
    pick_trans = AVD_transforms.PickInstances(chosen_ids,
                                              max_difficulty=4)
    #add more boxes in each image, each randomly perturbed from original
    perturb_trans = AVD_transforms.AddPerturbedBoxes(num_to_add=1,
                                                     changes = [[-.30,.40],
                                                               [-.30,.40],
                                                               [-.40,.30],
                                                               [-.40,.30]],
                                                    percents=True)


    #Make sure the boxes are valid(area>0, inside image)
    validate_trans =AVD_transforms.ValidateMinMaxBoxes(min_box_dimensions=[10,10],
                                                    image_dimensions=org_img_dims)
    #make class ids consecutive (see transforms docs)
    ids_trans = AVD_transforms.MakeIdsConsecutive(chosen_ids)
    #convert the labels to tensors
    to_tensor_trans = AVD_transforms.ToTensor()

    #compose the transforms in a specific order, first to last
    target_trans = AVD_transforms.Compose([
                                           pick_trans,
                                           perturb_trans,
                                           back_trans,
                                           validate_trans,
                                           ids_trans,
                                           to_tensor_trans])



    ##image transforms
    #normalize image to be [-1,1]
    norm_trans = AVD_transforms.NormalizePlusMinusOne()
    #resize the images to be image_size
    resize_trans = AVD_transforms.ResizeImage(image_size[0:2],'fill')

    #compose the image transforms in a specific order
    image_trans = AVD_transforms.Compose([resize_trans,
                                          to_tensor_trans,
                                          norm_trans,
                                         ])


    dataset = AVD.AVD(root=root,
                         scene_list=scene_list,
                         transform=image_trans,
                         target_transform=target_trans,
                         classification=classification,
                         preload_images=preload,
                         by_box=by_box)
    return dataset







def get_alexnet_AVD(root, scene_list,
                    preload=False,
                    detection=False):
    """
    Returns a dataloader for the AVD dataset for use with pretrained AlexNet.

    Uses get_standard_AVD, but changes image normalization to work with the 
    torch pretrained AlexNet model 
    dataloader = get_AlexNet_AVD('/path/to/data', ['scene1','scene2,...])


    ARGS:
        root: path to data. Parent of all scene directories
        scene_list: scenes to include
        
    KEYWORD ARGS:
        preload(False): if images should all be loaded at initialization
    """



    image_size=[224,224,3]


    #image transforms
    #normalize image to be [-1,1]
    norm_trans = AVD_transforms.NormalizeRange(0.0,1.0)
    norm_trans2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    #resize the images to be image_size
    resize_trans = AVD_transforms.ResizeImage(image_size[0:2],'fill')
    
    to_tensor_trans = AVD_transforms.ToTensor()



    #get dataset
    if detection:
        dataset = get_standard_AVD(root,scene_list,
                                   classification=False,
                                   by_box=False, 
                                   preload=preload)

    
        #resize the images to be image_size
        resize_trans = AVD_transforms.ResizeImage([1920/2,1080/2,3],'warp')

    else:
        dataset = get_standard_AVD(root,scene_list,
                                   classification=True,
                                   by_box=True, 
                                   preload=preload)




    #compose the image transforms in a specific order
    image_trans = AVD_transforms.Compose([resize_trans,
                                          to_tensor_trans,
                                          norm_trans,
                                          norm_trans2
                                         ])
    dataset.transform = image_trans 
    return dataset



def get_alexnet_classification_transform():
    image_size=[224,224,3]


    #image transforms
    #normalize image to be [-1,1]
    norm_trans = AVD_transforms.NormalizeRange(0.0,1.0)
    norm_trans2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    #resize the images to be image_size
    resize_trans = AVD_transforms.ResizeImage(image_size[0:2],'fill')

    to_tensor_trans = AVD_transforms.ToTensor()

    #compose the image transforms in a specific order
    image_trans = AVD_transforms.Compose([resize_trans,
                                          to_tensor_trans,
                                          norm_trans,
                                          norm_trans2
                                         ])
    return image_trans 






def get_fasterRCNN_AVD(root, scene_list,
                       preload=False,
                       max_difficulty=4,
                       chosen_ids=None,
                       by_box=False,
                       fraction_of_no_box=.1,
                      ):
    """
    Returns a dataloader for the AVD dataset for use with training FasterRCNN.

    dataset = get_fasterRCNN_AVD('/path/to/data', ['scene1','scene2,...])


    ARGS:
        root: path to data. Parent of all scene directories
        scene_list: scenes to include
        
    KEYWORD ARGS:
        preload(False): if images should all be loaded at initialization
        max_difficulty(int=4): max bbox difficulty to use 
    """
    if chosen_ids is None:
        chosen_ids = range(28)

    ##initialize transforms for the labels
    #only consider boxes from the chosen classes
    pick_trans = AVD_transforms.PickInstances(chosen_ids,
                                              max_difficulty=max_difficulty)
    to_tensor_trans = AVD_transforms.ToTensor()


    #compose the transforms in a specific order, first to last
    target_trans = AVD_transforms.Compose([
                                           pick_trans,
                                          ])


    ##image transforms
    means = np.array([[[102.9801, 115.9465, 122.7717]]])
    norm_trans = AVD_transforms.MeanSTDNormalize(mean=means)

    #compose the image transforms in a specific order
    image_trans = AVD_transforms.Compose([
                                          norm_trans,
                                          to_tensor_trans,
                                         ])

    id_to_name_dict = get_class_id_to_name_dict(root)

    dataset = AVD.AVD(root=root,
                         scene_list=scene_list,
                         transform=image_trans,
                         target_transform=target_trans,
                         classification=False,
                         preload_images=preload,
                         by_box=by_box,
                         class_id_to_name=id_to_name_dict,
                         fraction_of_no_box=fraction_of_no_box)

    return dataset



def get_fasterRCNN_AVD_to_orig_image_trans():

    means = np.array([[[-102.9801, -115.9465, -122.7717]]])
    denormalize = AVD_transforms.MeanSTDNormalize(mean=means)
    to_numpy = AVD_transforms.ToNumpy()

    return AVD_transforms.Compose([to_numpy, denormalize])



