import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD 
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
from PreDefinedSquareImageNet import PreDefinedSquareImageNet
import time
import numpy as np

data_path = '/playpen/ammirato/Data/RohitData/'
num_classes = 2
learning_rate = .001
batch_size = 32 
max_epochs = 5
image_size = [32,32,3]
cuda = True
chosen_ids = [0,5]
save_path = ('/playpen/ammirato/Documents/exploring_neural_networks/' + 
             'exploring_pytorch/saved_models/' + 'predefined.p')


#CREATE TRAIN/TEST splits
#label transforms
back_trans = AVD_transforms.AddBackgroundBoxes(num_to_add=2,
                                              box_dimensions_range=[100,100,200,200])
pick_trans = AVD_transforms.PickInstances(chosen_ids)
perturb_trans = AVD_transforms.AddPerturbedBoxes(num_to_add=7,
                                                changes = [[-50,10],
                                                           [-50,10], 
                                                           [-50,10], 
                                                           [-50,10]])
validate_trans =AVD_transforms.ValidateMinMaxBoxes()
ids_trans = AVD_transforms.MakeIdsConsecutive(chosen_ids)
to_tensor_trans = AVD_transforms.ToTensor()



target_trans = AVD_transforms.Compose([perturb_trans,
                                       back_trans,  
                                       pick_trans,
                                       validate_trans,
                                       ids_trans,
                                       to_tensor_trans])

#image transforms
norm_trans = AVD_transforms.NormalizePlusMinusOne()
resize_trans = AVD_transforms.ResizeImage(image_size[0:2],'fill')
image_trans = AVD_transforms.Compose([resize_trans,
                                      to_tensor_trans,
                                      norm_trans])

train_set = AVD.AVD_ByBox(root='/playpen/ammirato/Data/RohitData/',
                             scene_list=['Home_014_1',
                                         'Home_014_2',
                                         'Home_003_1',
                                         'Home_003_2',
                                         'Home_002_1'],
                             transform=image_trans,
                             target_transform=target_trans,
                             classification=True)
test_set = AVD.AVD_ByBox(root='/playpen/ammirato/Data/RohitData/',
                             scene_list=['Home_001_1','Home_001_2'],
                             transform=image_trans,
                             target_transform=target_trans,
                             classification=True)



trainloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,
                                          num_workers=2, collate_fn=AVD.collate)
testloader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,
                                          num_workers=2, collate_fn=AVD.collate)



#Define model, optimizer, and loss function
model = PreDefinedSquareImageNet(image_size,num_classes)
model.load_state_dict(torch.load(save_path))
if cuda:
    model.cuda()

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
loss_fn = torch.nn.NLLLoss()



#TRAIN LOOP
for epoch in range(max_epochs):
    for il,data in enumerate(trainloader):


        batch_imgs,batch_labels = data
        if cuda:
            batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda() 


        optimizer.zero_grad()     
        y_pred = model(Variable(batch_imgs))
        loss = loss_fn.forward(y_pred,Variable(batch_labels))

        loss.backward()
        optimizer.step()
        


        if il % 10 == 0:
            print 'Iter: ' + str(il) + '  Loss : ' + str(loss.data[0])    



    num_correct = 0
    total = 0
    for il, data in enumerate(testloader):

        batch_imgs,batch_labels = data
        if cuda:
            batch_imgs = batch_imgs.cuda()
   
        y_pred = model(Variable(batch_imgs))
        pred_class = torch.LongTensor([np.argmax(el) for el in y_pred.cpu().data.numpy()])  
        diff = batch_labels - pred_class

        num_correct += len(diff) - np.count_nonzero(diff.numpy())
        total += len(diff) 
        if total > 100:
            break

    print 'Test: Correct: {}  Total: {}  Accuracy: {}'.format(
                                            num_correct,
                                            total,
                                            num_correct/float(total))


    print '\n\nEpoch :  ' + str(epoch) + '\n\n'


torch.save(model.state_dict(), save_path)

