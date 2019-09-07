# The newest OpenPose Pytorch Implementation
# __author__ = "Haiyang Liu"

import torch
import torch.nn as nn
from torch.nn import init

HEATMAP_NUM = 19
PAF_NUM = 38

class CMUnetwork(nn.Module):
    ''' the newest cmu network'''
    def __init__ (self):
        # already finish the init_weight in each block
        self.state_0 = VGG_block()
        self.state_1 = state_n_block(128, PAF_NUM)
        self.state_2 = state_n_block(128+PAF_NUM,PAF_NUM)
        self.state_3 = state_n_block(128+PAF_NUM,PAF_NUM)
        self.state_4 = state_n_block(128+PAF_NUM,PAF_NUM)
        self.state_5 = state_n_block(128+PAF_NUM,PAF_NUM)
        self.state_6 = state_n_block(128+PAF_NUM,HEATMAP_NUM)

    def forward(self,input_0):

        saved_for_loss = []

        output_0 = self.state_0(input_0)
        output_1 = self.state_1(output_0)
        input_2  = torch.cat([output_1,output_0],1)
        saved_for_loss.append(output_1)

        output_2 = self.state_2(input_2)
        input_3  = torch.cat([output_2,output_0],1)
        saved_for_loss.append(output_2)

        output_3 = self.state_3(input_3)
        input_4  = torch.cat([output_3,output_0],1)
        saved_for_loss.append(output_3)

        output_4 = self.state_4(input_4)
        input_5  = torch.cat([output_4,output_0],1)
        saved_for_loss.append(output_4)

        output_5 = self.state_5(input_5)
        input_6  = torch.cat([output_5,output_0],1)
        saved_for_loss.append(output_5)

        output_6 = self.state_6(input_6)
        saved_for_loss.append(output_6)

        return output_6, saved_for_loss


class dense_block(nn.Module):
    '''1. basic dense block of the new openpose
       2. add conv1,2,3 output together to output
       3. default kernal_size = 3,bias = true
    '''
    def __init__(self, in_dim, out_dim):
        # default inplace = False for ReLU
        self.conv1 = nn.Sequential([nn.Conv2d(in_dim, 128, 3, 1, 1),
                                    nn.ReLU(inplace=True)])
        self.conv2 = nn.Sequential([nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.ReLU(inplace=True)])
        self.conv3 = nn.Sequential([nn.Conv2d(128, (out_dim-256), 3, 1, 1),
                                    nn.ReLU(inplace=True)])
        self.initialize_weight()
        

    def forward(self,input_1):
        output_1 = self.conv1(input_1)
        output_2 = self.conv2(output_1)
        output_3 = self.conv3(output_2)
        output = torch.cat([output_1,output_2,output_3],1)
        return output
    
    def initialize_weight(self):
        for m in self.modules():
            print('need check init')
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    
        
class state_n_block(nn.Module):
    ''' 1. for state 1 in_dim = 128
        2. for other state depend on the paf and heatmap channels
    '''
    def __init__(self, in_dim, out_dim):
        # 384 = 128 *3
        self.block1 = dense_block(in_dim,384)
        self.block2 = dense_block(384,384)
        self.block3 = dense_block(384,384)
        self.block4 = dense_block(384,384)
        self.block5 = dense_block(384,384)
        self.conv1  = nn.Sequential([nn.Conv2d(384, 512, 1, 1, 0),
                                     nn.ReLU(inplace = True)])
        self.conv2  = nn.Conv2d(512,out_dim,1,1,0)
        self.initialize_weight()

    def forward(self,input_1):
        ''' inplace the midresult '''
        output_1 = self.block1(input_1)
        output_1 = self.block2(output_1)
        output_1 = self.block3(output_1)                     
        output_1 = self.block4(output_1)
        output_1 = self.block5(output_1)
        output_1 = self.conv1(output_1)
        output_1 = self.conv2(output_1)
        return output_1
    
    def initialize_weight(self):
        '''init 1*1 conv block
        '''
        init.normal_(self.conv1[1].weight, std =0.01)
        init.constant_(self.conv1[1].bias, 0.0)
        init.normal_(self.conv2.weight, std =0.01)
        init.constant_(self.conv2.bias, 0.0)


class VGG_block(nn.Module):
    ''' 1. default have the bias
        2. using relu and 3 * max pooling
        3. 10 layers of VGG original
        4. 2 extra layers by CMU
        5. default in_dim = 3,out_dim = 128
        6. all kernal_size = 3, stride = 1
    '''
    def __init__(self, in_dim = 3, out_dim = 128):
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool_1 = nn.Maxpooling(2, 2, 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.pool_2 = nn.Maxpooling(2, 2, 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pool_3 = nn.Maxpooling(2, 2, 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3_cmu = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv4_4_cmu = nn.Conv2d(256, 128, 3, 1, 1)
        self.initialize_weight()
                                
    def forward(self,input_1):
        '''inplace middle result '''
        output_1 = self.block1.forward(input_1)
        output_1 = self.block2.forward(output_1)
        output_1 = self.block3.forward(output_1)                     
        output_1 = self.block4.forward(output_1)
        output_1 = self.block5.forward(output_1)
        output_1 = self.conv1(output_1)
        output_1 = self.conv2(output_1)
        return output_1
    
    def initilization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  
                    init.constant_(m.bias, 0.0)

 


    






    
