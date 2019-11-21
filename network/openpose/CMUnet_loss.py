# ------------------------------------------------------------------------------
# define the CMUnet loss calculation 
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

import torch 
import torch.nn as nn



def loss_cli(parser,name):
    print('using',name,'loss success')
    group = parser.add_argument_group('loss')
    group.add_argument('--auto_weight', default=False, type=bool)

def get_offset_loss(saved_for_loss,target_heat,heat_mask,target_paf,paf_mask,target_offset,args,epoch):
    ''' input： the output of CMU net
                the target img
                the mask for unanno-file
                config control the weight of loss
    '''
    loss = {}
    loss['final'] = 0
    batch_size = args.batch_size
    criterion = My_loss().cuda()
    criterion_offset = My_loss_offset().cuda()
    heat_output = saved_for_loss[-2]
    heat_output_copy = torch.zeros(heat_output.shape)
    heat_output_copy.copy_(heat_output)
    heat_output_copy = heat_output_copy.cuda()
    heat_output_copy_final = torch.zeros([heat_output.shape[0],heat_output.shape[1]*2,
                                    heat_output.shape[2],heat_output.shape[3]])
    for i in range(heat_output.shape[1]):
        heat_output_copy_final[:,2*i,:,:] = heat_output_copy[:,i,:,:]
        heat_output_copy_final[:,2*i+1,:,:] = heat_output_copy[:,i,:,:]
    heat_output_copy_final = heat_output_copy_final.cuda()
    # for debug
    # print(target_heat.size())
    # print(heat_mask.size())
    # print(target_paf.size())
    # print(paf_mask.size())
    # print(saved_for_loss[0].size())
    # print(saved_for_loss[1].size())

    for i in range(args.paf_stage):

        loss['stage_{}'.format(i)] = criterion(saved_for_loss[i] * paf_mask,target_paf * paf_mask,batch_size)
        loss['final'] += loss['stage_{}'.format(i)]
    for i in range(args.paf_stage,6):
        loss['stage_{}'.format(i)] = criterion(saved_for_loss[i] * heat_mask,target_heat  * heat_mask,batch_size)
        loss['final'] += loss['stage_{}'.format(i)] 
    for i in range(6,7):
        loss['stage_{}'.format(i)] = criterion_offset(saved_for_loss[-1], heat_output_copy_final,target_offset,batch_size)
        if epoch > 4:
            loss['final'] += loss['stage_{}'.format(i)]
    return loss



def get_loss(saved_for_loss,target_heat,target_paf,args,wei_con):
    '''
    input： the output of CMU net
            the target img
            the mask for unanno-file
            onfig control the weight of loss
    '''
    loss = {}
    length = len(saved_for_loss)
    loss['final'] = 0
    
    weights = torch.ones([6,args.paf_num+args.heatmap_num])
    
    #print("weights size",weights.size())
    #print("weigcon size",wei_con.size())
    if args.auto_weight == True:
        for i in range(args.paf_num+args.heatmap_num):
            weights[0][i] = wei_con[0][i]
    weights = weights.cuda()
    
    criterion = nn.MSELoss(size_average=True).cuda()

    if args.auto_weight == True:
        for i in range(args.paf_stage):
            loss['stage_{}'.format(i)] = 0
            for j in range(args.paf_num):
                #print(saved_for_loss[i].size(),target_paf.size())
                #print(weights.size())
                #loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],target_paf[:,j,:,:]) 
                loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],target_paf[:,j,:,:]) * weights[0][j]
                loss['stage_{}'.format(i)] += loss['stage_{0}_{1}'.format(i,j)]
            loss['stage_{}'.format(i)] /= 38
            loss['final'] += loss['stage_{}'.format(i)]

        for i in range(args.paf_stage,length):
            loss['stage_{}'.format(i)] = 0
            for j in range(args.heatmap_num):
                loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],target_heat[:,j,:,:]) * weights[0][j+args.paf_num]
                loss['stage_{}'.format(i)] += loss['stage_{0}_{1}'.format(i,j)]
            loss['stage_{}'.format(i)] /= 19
            loss['final'] += loss['stage_{}'.format(i)]
    
    else:
        for i in range(args.paf_stage):
            loss['stage_{}'.format(i)] = 0
            for j in range(args.paf_num):
                #print(saved_for_loss[i].size(),target_paf.size())
                #print(weights.size())
                loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],target_paf[:,j,:,:]) 
                loss['stage_{}'.format(i)] += loss['stage_{0}_{1}'.format(i,j)]
            loss['stage_{}'.format(i)] /= 38
            loss['final'] += loss['stage_{}'.format(i)]

        for i in range(args.paf_stage,length):
            loss['stage_{}'.format(i)] = 0
            for j in range(args.heatmap_num):
                loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],target_heat[:,j,:,:])
                loss['stage_{}'.format(i)] += loss['stage_{0}_{1}'.format(i,j)]
            loss['stage_{}'.format(i)] /= 19
            loss['final'] += loss['stage_{}'.format(i)]
    
    return loss

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, batch_size):
        return torch.sum(torch.pow((x - y), 2))/batch_size/2

class My_loss2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, batch_size,mask):
        return torch.sum(torch.pow((x - y), 2) * mask)/batch_size/2

class My_loss_offset(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, mask, y, batch_size):
        return torch.sum(torch.pow((x - y), 2) * mask)/batch_size/2
        
def get_old_loss(saved_for_loss,target_heat,target_paf,args,wei_con):
   
    loss = {}
    loss['final'] = 0
    batch_size = args.batch_size
    
    criterion = nn.MSELoss(size_average=True)
    for i in range(6):
        loss['stage_1_{}'.format(i)] = criterion(saved_for_loss[2*i],target_paf,batch_size)
        loss['stage_2_{}'.format(i)] = criterion(saved_for_loss[2*i+1],target_heat,batch_size)
        loss['final'] += loss['stage_1_{}'.format(i)]
        loss['final'] += loss['stage_2_{}'.format(i)] 
    return loss

def get_mask_loss(saved_for_loss,target_heat,heat_mask,target_paf,paf_mask,args,wei_con):
    ''' input： the output of CMU net
                the target img
                the mask for unanno-file
                config control the weight of loss
    '''
    loss = {}
    loss['final'] = 0
    batch_size = args.batch_size
    criterion = My_loss().cuda()
    # for debug
    # print(target_heat.size())
    # print(heat_mask.size())
    # print(target_paf.size())
    # print(paf_mask.size())
    # print(saved_for_loss[0].size())
    # print(saved_for_loss[1].size())

    for i in range(6):

        loss['stage_1_{}'.format(i)] = criterion(saved_for_loss[2*i] * paf_mask,target_paf * paf_mask,batch_size)
        loss['stage_2_{}'.format(i)] = criterion(saved_for_loss[2*i+1] * heat_mask,target_heat  * heat_mask,batch_size)
        loss['final'] += loss['stage_1_{}'.format(i)]
        loss['final'] += loss['stage_2_{}'.format(i)] 
    return loss

def get_old_loss(saved_for_loss,target_heat,target_paf,args,wei_con):
   
    loss = {}
    loss['final'] = 0
    batch_size = args.batch_size
    
    criterion = nn.MSELoss(size_average=True)
    for i in range(6):
        loss['stage_1_{}'.format(i)] = criterion(saved_for_loss[2*i],target_paf,batch_size)
        loss['stage_2_{}'.format(i)] = criterion(saved_for_loss[2*i+1],target_heat,batch_size)
        loss['final'] += loss['stage_1_{}'.format(i)]
        loss['final'] += loss['stage_2_{}'.format(i)] 
    return loss

def get_mask_loss_self(saved_for_loss,target_heat,heat_mask,target_paf,paf_mask_self,args,wei_con):
    ''' input： the output of CMU net
                the target img
                the mask for unanno-file
                config control the weight of loss
    '''
    loss = {}
    loss['final'] = 0
    batch_size = args.batch_size
    criterion = My_loss().cuda()
    criterion2 = My_loss2().cuda()

    factors = 1
    # for debug
    # print(target_heat.size())
    # print(heat_mask.size())
    # print(target_paf.size())
    # print(paf_mask.size())
    # print(saved_for_loss[0].size())
    # print(saved_for_loss[1].size())
    for i in range(6):

        loss['stage_1_{}'.format(i)] = factors * criterion2(saved_for_loss[2*i] * paf_mask_self[:19,:,:],target_paf * paf_mask_self[:19,:,:],batch_size,paf_mask_self[19:,:,:])
        loss['stage_2_{}'.format(i)] = criterion(saved_for_loss[2*i+1] * heat_mask,target_heat  * heat_mask,batch_size)
        loss['final'] += loss['stage_1_{}'.format(i)]
        loss['final'] += loss['stage_2_{}'.format(i)] 
    return loss


def get_new_mask_loss(saved_for_loss,target_heat,heat_mask,target_paf,paf_mask,args,wei_con):
    ''' input： the output of CMU net
                the target img
                the mask for unanno-file
                config control the weight of loss
    '''
    loss = {}
    loss['final'] = 0
    batch_size = args.batch_size
    criterion = My_loss().cuda()
    # for debug
    # print(target_heat.size())
    # print(heat_mask.size())
    # print(target_paf.size())
    # print(paf_mask.size())
    # print(saved_for_loss[0].size())
    # print(saved_for_loss[1].size())

    for i in range(args.paf_stage):
        loss['stage_{}'.format(i)] = criterion(saved_for_loss[i] * paf_mask,target_paf * paf_mask,batch_size)
        loss['final'] += loss['stage_{}'.format(i)]
    for i in range(args.paf_stage,6):
        loss['stage_{}'.format(i)] = criterion(saved_for_loss[i] * heat_mask,target_heat  * heat_mask,batch_size)
        loss['final'] += loss['stage_{}'.format(i)] 
    return loss