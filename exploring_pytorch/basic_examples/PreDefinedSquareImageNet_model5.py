import torch



class Flatten(torch.nn.Module):
    """
    Flattens a 4D tensor to 2D.  

    initialize with the length of the 2D tensor. The 2D tensor will be:
    (batch_size x length) = number of elements in 4D tensor. That 
    equality must hold true
    """

    def __init__(self, length2d):
        super(Flatten,self).__init__()
        self.length2d = length2d

    def forward(self,input):
        return input.view([-1,self.length2d])





class PreDefinedSquareImageNet(torch.nn.Module):
    """
    Makes a neural network, must see code to see model(only 2 custom inputs).

    Takes custom image_size and number of classes 
    ex) model = DefinedNet(image_size,num_classes)

    ARGS:
        image_size (List[int,int,int]) = height, width, num channels
                                         MUST BE SQUARE IMAGE
                                         (height = width)
        num_classes (int) 
    """


    def __init__(self, image_size, num_classes):
        super(PreDefinedSquareImageNet, self).__init__()
       
        #must be square image 
        assert(image_size[0] == image_size[1])

        num_channels = image_size[2]
        image_size = image_size[0]

        #add layers, and keep track of image size after each layer
        layers = []

        #conv1, with max pool
        #layers.append(torch.nn.Conv2d(3,5,5))
        #image_size = image_size - (5-1)
        #layers.append(torch.nn.ReLU())
        #image_size = image_size
        #layers.append(torch.nn.MaxPool2d((2,2)))
        #image_size = int(image_size/2)

        #conv2, with max pool
        #layers.append(torch.nn.Conv2d(5,5,5))
        #image_size = image_size - (5-1)
        #layers.append(torch.nn.ReLU())
        #image_size = image_size
        #layers.append(torch.nn.MaxPool2d((2,2)))
        #image_size = int(image_size/2)

        #fully connected layer
        #layers.append(Flatten(image_size*image_size*5))
        layers.append(Flatten(image_size*image_size*3))
        #layers.append(torch.nn.Linear(image_size*image_size*3, 256))
        layers.append(torch.nn.Linear(image_size*image_size*3,num_classes))
        layers.append(torch.nn.LogSoftmax())

        #put the layers in the net
        #must do so model parameters are accesable for
        #autograd/optimizer on backward pass/update
        idx = 0
        for module in layers:
            self.add_module(str(idx),module) 
            idx +=1 

    def forward(self,input):
        for module in self._modules.values():
            input = module(input)
        return input 



    def forward_vis(self,input):
        vis_outputs = []
        for module in self._modules.values():
            input = module(input)
            vis_outputs.append(input)

        return input, vis_outputs
