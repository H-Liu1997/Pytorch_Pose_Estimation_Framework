#define the CMUnet loss calculation
#__author__ = 'Haiyang Liu'
import numpy as np
import torch.nn as nn



def cli(parser):
    group = parser.add_argument_group('loss')
    group.add_argument('--auto_weight', default=False, type=bool)
    group.add_argument('--weight_save_train_dir', default="xxx")
    group.add_argument('--weight_save_val_dir', default="xxx")
    group.add_argument('--weight_vgg19', default='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
    group.add_argument('--lr', default=1., type=float)
    group.add_argument('--weight_decay', default=0., type=float)
    group.add_argument('--momentum', default=0.9, type=float)
    group.add_argument('--nesterov', default=True, type=bool)


def get_loss(saved_for_loss,target,args,wei_con):
    ''' input： the output of CMU net
                the target img
                the mask for unanno-file
                config control the weight of loss
    '''
    loss = {}
    length = len(saved_for_loss)
    
    
    weights = np.ones(1,args.paf_num + args.heatmap_num)
    weights = weights.cuda()
    if args.auto_weight == True:
        for i in range(args.paf_num+args.heatmap_num):
            weights[i] = wei_con[i]
        
    target_paf = target[:,:args.paf_num-1,:,:]
    target_heat = target[:,args.paf_num:args.paf_num+args.heatmap_num-1,:,:]
    gt_paf = target_paf 
    gt_heat = target_heat 
    criterion = nn.MSELoss(size_average=True).cuda()
  
    for i in range(args.paf_stage):
        for j in range(args.paf_num):
            loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],gt_paf[:,j,:,:]) * weights[j]
            loss['stage_{}'.format(i)] += loss['stage_{0}_{1}'.format(i,j)]
        loss['final'] += loss['stage_{}'.format(i)]

    for i in range(args.paf_stage,length):
        for j in range(args.heatmap_num):
            loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],gt_heat[:,j,:,:]) * weights[j+args.paf_num]
            loss['stage_{}'.format(i)] += loss['stage_{0}_{1}'.format(i,j)]
        loss['final'] += loss['stage_{}'.format(i)]
    
    return loss['final'],loss