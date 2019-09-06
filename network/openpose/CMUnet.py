# The newest OpenPose Pytorch Implementation
# __author__ = "Haiyang Liu"

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import init

HEATMAP_NUM = 19
PAF_NUM = 38

class CMUnetwork(nn.net):
    def __init__ (self, ):
        self.state_0 = VGG_block()
        self.state_1 = state_n_block()
        self.state_2 = state_n_block()
        self.state_3 = state_n_block()
        self.state_4 = state_n_block()
        self.state_5 = state_n_block()
        self.state_6 = state_n_block()

    def forward(self):
        pass

    def initilization(self):
        pass


class dense_block(nn.net):
    '''basic dense block of the new openpose
       add conv1,2,3 output together to output
       default kernal_size = 3
    '''
    def __init__(self, in_dim, out_dim, kernal_size, stride, padding, bias = True):
        self.conv1 = nn.Conv2d(in_dim, 128, kerner_size = kernal_size,
                               stride = stride, padding = padding, bias = bias)
        self.conv2 = nn.Conv2d(128, 128, kerner_size = kernal_size,
                               stride = stride, padding = padding, bias = bias)
        self.conv3 = nn.Conv2d(128, (out_dim-128-128), kerner_size = kernal_size,
                               stride = stride, padding = padding, bias = bias)

    def forward(self,input_1):
        output_1 = self.conv1(input_1)
        output_2 = self.conv2(output_1)
        output_3 = self.conv3(output_2)
        output = torch.cat([output_1,output_2,output_3],1)
        return output
    
    def initilization(self):
        pass

class state_n_block(nn.net):
    def __init__(self, in_dim, out_dim, last_Paf_or_heat, Paf_or_heat, bias = True):
        if last_Paf_or_heat == 'paf':
            self.block1 = dense_block(128+PAF_NUM,128*3,3,1,1)
        elif last_Paf_or_heat == 'heatmap':
            self.block1 = dense_block(128+HEATMAP_NUM,128*3,3,1,1)
        else:
            self.block1 = dense_block(128,128*3,3,1,1)

        self.block2 = dense_block(128*3,128*3,3,1,1)
        self.block3 = dense_block(128*3,128*3,3,1,1)
        self.block4 = dense_block(128*3,128*3,3,1,1)
        self.block5 = dense_block(128*3,128*3,3,1,1)

        self.conv1  = nn.Conv2d(128*3, 512, kerner_size = 1,
                               stride = 1, padding = 0, bias = True)
        if Paf_or_heat == 'paf':
            self.conv2  = nn.Conv2d(512, PAF_NUM, kerner_size = 1,
                                stride = 1, padding = 0, bias = True)
        else:
            self.conv2  = nn.Conv2d(512, HEATMAP_NUM, kerner_size = 1,
                                stride = 1, padding = 0, bias = True)

    def forward(self,input_1):
        output_1 = self.block1.forward(input_1)
        output_1 = self.block2.forward(output_1)
        output_1 = self.block3.forward(output_1)                     
        output_1 = self.block4.forward(output_1)
        output_1 = self.block5.forward(output_1)
        output_1 = self.conv1(output_1)
        output_1 = self.conv2(output_1)
        return output_1
    
    def initilization(self):
        pass


class VGG_block(nn.net):
    def __init__(self, in_dim, out_dim, bias = True):
        self.conv1 = nn.Conv2d(3, 64, 1,)
        self.block2 = dense_block(128*3,128*3,3,1,1)
        self.block3 = dense_block(128*3,128*3,3,1,1)
        self.block4 = dense_block(128*3,128*3,3,1,1)
        self.block5 = dense_block(128*3,128*3,3,1,1)
        self.conv1  = nn.Conv2d(128*3, 512, kerner_size = 1,)
                                
    def forward(self,input_1):
        output_1 = self.block1.forward(input_1)
        output_1 = self.block2.forward(output_1)
        output_1 = self.block3.forward(output_1)                     
        output_1 = self.block4.forward(output_1)
        output_1 = self.block5.forward(output_1)
        output_1 = self.conv1(output_1)
        output_1 = self.conv2(output_1)
        return output_1
    
    def initilization(self):
        pass

 


    






    
