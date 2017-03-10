import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable


#load the image using PIL
pil_img = Image.open('/playpen/ammirato/Pictures/cola.jpg')
#convert image to numpy
np_img = np.asarray(pil_img) 
#reshape the numpy image to put channel first
#rs_np_img = np_img.reshape(np_img.shape[2],np_img.shape[0],np_img.shape[1])
rs_np_img = np.transpose(np_img,(2,0,1))

#convert both images to tensors, add batch dimension
to_tensor = transforms.ToTensor()
pil_tensor = to_tensor(pil_img).unsqueeze(0)
np_tensor = to_tensor(rs_np_img).unsqueeze(0)

#create a convolution layer, 3 input channels, 4 output, 5x5 filters
conv = torch.nn.Conv2d(3,4,5)

#run the images through the filter
pil_filtered = conv(Variable(pil_tensor))
np_filtered = conv(Variable(np_tensor))

#see if the results are different
diff = pil_filtered.data - np_filtered.data
print diff.mean()
print 'PIL max val: ' + str(pil_filtered.data.max())
print 'NP max val: ' + str(np_filtered.data.max())


