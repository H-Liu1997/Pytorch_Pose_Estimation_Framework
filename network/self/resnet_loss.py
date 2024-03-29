# ------------------------------------------------------------------------------
# define the CMUnet loss calculation 
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------
import torch 
import torch.nn as nn



def loss_cli(parser,name):
    print('choose ',name,'loss success')
    group = parser.add_argument_group('loss')
    group.add_argument('--auto_weight', default=False, type=bool)


def get_loss(saved_for_loss,target_heat,target_paf,args,wei_con):
    ''' input： the output of CMU net
                the target img
                the mask for unanno-file
                config control the weight of loss
    '''
    loss = {}
    loss['final'] = 0
    
    # weights = torch.ones([6,args.paf_num+args.heatmap_num])
    
    # #print("weights size",weights.size())
    # #print("weigcon size",wei_con.size())
    # if args.auto_weight == True:
    #     for i in range(args.paf_num+args.heatmap_num):
    #         weights[0][i] = wei_con[0][i]
    # weights = weights.cuda()
    
    criterion = nn.MSELoss(size_average=True).cuda()

    #if args.auto_weight == True:
        # for i in range(args.paf_stage):
        #     loss['stage_{}'.format(i)] = 0
        #     for j in range(args.paf_num):
        #         #print(saved_for_loss[i].size(),target_paf.size())
        #         #print(weights.size())
        #         #loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],target_paf[:,j,:,:]) 
        #         loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],target_paf[:,j,:,:]) * weights[0][j]
        #         loss['stage_{}'.format(i)] += loss['stage_{0}_{1}'.format(i,j)]
        #     loss['stage_{}'.format(i)] /= 38
        #     loss['final'] += loss['stage_{}'.format(i)]

        # for i in range(args.paf_stage,length):
        #     loss['stage_{}'.format(i)] = 0
        #     for j in range(args.heatmap_num):
        #         loss['stage_{0}_{1}'.format(i,j)] = criterion(saved_for_loss[i][:,j,:,:],target_heat[:,j,:,:]) * weights[0][j+args.paf_num]
        #         loss['stage_{}'.format(i)] += loss['stage_{0}_{1}'.format(i,j)]
        #     loss['stage_{}'.format(i)] /= 19
        #     loss['final'] += loss['stage_{}'.format(i)]
    
    #else:
    for i in range(3):
        loss['stage_{}'.format(i)] = criterion(saved_for_loss[i],target_paf)
        loss['final'] += 0.5 * loss['stage_{}'.format(i)]
    
    for i in range(3,4):

        loss['stage_{}'.format(i)] = criterion(saved_for_loss[i],target_heat) 
        loss['final'] += 0.5 * loss['stage_{}'.format(i)] 
    return loss