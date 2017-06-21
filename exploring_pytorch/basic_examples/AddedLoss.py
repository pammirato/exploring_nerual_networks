import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F


class AddedLoss(torch.nn.modules.loss._Loss):


    def forward(self,input,target):
        _assert(no_grad_target)

        return (F.cross_entropy(input,target,self.weight,self.size_average) + 
                F.cross_entropy(input,target)) 
