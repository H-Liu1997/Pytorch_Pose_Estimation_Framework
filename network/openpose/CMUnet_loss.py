#define the CMUnet loss calculation
#__author__ = 'Haiyang Liu'
import numpy as np
import torch.nn as nn

HEATMAP_NUM = 19
PAF_NUM = 38

def get_loss(saved_for_loss,target,mask,config):
    ''' input： the output of CMU net
                the target img
                the mask for unanno-file
                config control the weight of loss
    '''
    loss = {}
    length = len(saved_for_loss)

    target_paf = target[:,:PAF_NUM-1,:,:]
    target_heat = target[:,PAF_NUM:PAF_NUM+HEATMAP_NUM-1,:,:]
    mask_paf = mask[:,:PAF_NUM-1,:,:]
    mask_heat = mask[:,PAF_NUM:PAF_NUM+HEATMAP_NUM-1,:,:]
    gt_paf = target_paf * mask_paf
    gt_heat = target_heat * mask_heat
    criterion = nn.MSELoss(size_average=True).cuda()

    for i in range(config['loss']['paf_num']):
        loss['stage_{}'.format(i)] = criterion(saved_for_loss[i],gt_paf)
        loss['final'] += loss['stage_{}'.format(i)]
    for i in range(config['loss']['paf_num'],length):
        loss['stage_{}'.format(i)] = criterion(saved_for_loss[i],gt_heat)
        loss['final'] += loss['stage_{}'.format(i)]
    
    return loss['final'],loss