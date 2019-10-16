# ------------------------------------------------------------------------------
# The train code of total framework 
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

import time
import argparse
from collections import OrderedDict
import json
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.model_zoo as model_zoo
import numpy as np
from tensorboardX import SummaryWriter

from .datasets import h5loader
from .network.openpose import CMUnet_loss, CMU_old,rtpose_vgg_
#from .network import loss_factory,net_factory
from . import evaluate


def cli():
    """
    setting all parameters
    1. hyper-parameters of building a network is in network.cli
    2. loss control parameters is in loss.cli
    3. data loader parameters such as path is in loader.cli
    4. evaluate threshold in val_cli 
    5. basic hyper-para such as learnling rate is in this file
    """
    print("if you change the network, make sure:","\n",
          "1.changing the network_cli","\n",
          "2.changing the net loader","\n",
          "3.changing loss in train and val")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--name',           default='op_repose11_no_decay',  type=str)
    parser.add_argument('--net',            default='CMU_old',  type=str)
    parser.add_argument('--loss',           default='mask_after_gt_alreadycheck',  type=str)
    parser.add_argument('--loader',         default='CMU',            type=str)
    parser.add_argument('--multi_lr',       default='4 kinds use',            type=str)
    parser.add_argument('--bias_decay',     default='use 0 for bias',            type=str)
    parser.add_argument('--pre_',           default='rtpose',            type=str)

    CMU_old.network_cli(parser)
    CMUnet_loss.loss_cli(parser)
    #loader_factory.loader_cli(parser,"CMU")
    evaluate.val_cli(parser)
    
    # trian setting
    parser.add_argument('--pre_train',      default=0,          type=int)
    parser.add_argument('--freeze_base',    default=0,          type=int,       help='number of epochs to train with frozen base')
    parser.add_argument('--epochs',         default=300,        type=int)
    parser.add_argument('--per_batch',      default=5,         type=int,       help='batch size per gpu')
    parser.add_argument('--gpu',            default=[0,1],        type=list,      help="gpu number")
    
    # optimizer
    parser.add_argument('--opt_type',       default='sgd',      type=str,       help='sgd or adam')
    parser.add_argument('--lr',             default=2e-5,       type=float)
    parser.add_argument('--w_decay',        default=5e-4,       type=float)
    parser.add_argument('--beta1',          default=0.90,       type=float)
    parser.add_argument('--beta2',          default=0.999,      type=float)
    parser.add_argument('--nesterov',       default=False,      type=bool,      help='for sgd')

    parser.add_argument('--auto_lr',        default=True,       type=bool,      help='using auto lr control or not')
    parser.add_argument('--lr_tpye',        default='ms',       type=str,       help='milestone or auto_val')
    parser.add_argument('--factor',         default=0.333,      type=float,     help='divide factor of lr')
    parser.add_argument('--patience',       default=3,          type=int)
    parser.add_argument('--step',           default=[17,34,51,68],       type=list)

    # others
    parser.add_argument('--log_base',       default="./Pytorch_Pose_Estimation_Framework/ForSave/log/")
    parser.add_argument('--weight_pre',     default="./Pytorch_Pose_Estimation_Framework/ForSave/weight/pretrain/")
    parser.add_argument('--weight_base',    default="./Pytorch_Pose_Estimation_Framework/ForSave/weight/")
    parser.add_argument('--checkpoint',     default="./Pytorch_Pose_Estimation_Framework/ForSave/weight/op_repose11_no_/train_final.pth")
    parser.add_argument('--print_fre',      default=5,          type=int)
    parser.add_argument('--val_type',       default=0,          type=int)
    
    args = parser.parse_args()
    return args


def main():
    '''load config parameters'''
    args = cli()
    save_config(args)
    
    '''data portion'''
    #train_factory = loader_factory.loader_factory(args)
    train_loader = h5loader.train_factory('train',args)
    val_loader = h5loader.train_factory('val',args)
    
    '''network portion'''
    
    #model = CMU_old.CMUnetwork(args)
    # network = net_factory.net_factory(args.net)
    # model = network(args)
    # multi_gpu and cuda, will occur some bug when inner some function
    model = rtpose_vgg_.get_model(trunk='vgg19')
    #model = encoding.nn.DataParallelModel(model, device_ids=args.gpu_ids)
    #model = torch.nn.DataParallel(model).cuda()
    # load pretrained
    #rtpose_vgg_.use_vgg(model, args.weight_pre, 'vgg19')

    model = torch.nn.DataParallel(model,args.gpu).cuda()
    optimizer,lr_scheduler = optimizer_settings(False,model,args)
    start_epoch = load_checkpoints(model,optimizer,lr_scheduler,args)
    #start_epoch = 0

    '''val loss boundary and tensorboard path'''
    val_loss_min = np.inf
    lr = args.lr
    writer = SummaryWriter(args.log_path)
    flag = 0

    '''start freeze training'''
    if args.freeze_base != 0 and start_epoch <= args.freeze_base:
        flag = 1
        print("start freeze some weight training for epoch {}-{}".format(start_epoch,args.freeze_base)) 
        optimizer,lr_scheduler = optimizer_settings(True,model,args)
        
        for epoch in range(start_epoch,args.freeze_base):
            loss_train = train_one_epoch(train_loader,model,optimizer,writer,epoch,args)
            loss_val, accuracy_val = val_one_epoch(val_loader,model,epoch,args)
            '''save to tensorboard'''
            writer.add_scalars('train_val_loss_epoch', {'train loss': loss_train,
                                                  'val loss': loss_val}, epoch)
            writer.add_scalar('accuracy_epoch', accuracy_val, epoch)
            writer.add_scalar('lr_epoch', lr, epoch)
            
            val_loss_min = save_checkpoints(model,optimizer,lr_scheduler,epoch,loss_val,val_loss_min,args)
    
    '''start normal training'''
    print("start normal training")
    if flag: 
        optimizer,lr_scheduler = optimizer_settings(False,model,args)
        start_epoch = args.freeze_base
    for epoch in range(start_epoch,args.epochs):
        loss_train = train_one_epoch(train_loader,model,optimizer,writer,epoch,args)
        loss_val, accuracy_val = val_one_epoch(val_loader,model,epoch,args)

        if args.auto_lr:
            lr_scheduler.step(loss_val)
        else:
            print('no using lr_scheduler')

        '''save to tensorboard'''
        writer.add_scalars('train_val_loss', {'train loss': loss_train,
                                                'val loss': loss_val}, epoch)
        writer.add_scalar('accuracy', accuracy_val, epoch)
        writer.add_scalar('lr_epoch', lr, epoch)

        val_loss_min = save_checkpoints(model,optimizer,lr_scheduler,epoch,loss_val,val_loss_min,args)
    writer.close()


def save_config(args):
    """
    save the parameters to a txt file in the logpath
    1. contains all hyper-parameters of training 
    """
    batch_size = len(args.gpu) * args.per_batch
    args.log_path = os.path.join(args.log_base,args.name)
    args.weight_path = os.path.join(args.weight_base,args.name)
    args.batch_size = batch_size
    try:
        os.mkdir(args.log_path)
        print("create log save file")
        
    except:
        print('already exist the log file, please remove them if needed')
        try:
            os.mkdir(args.weight_path)
            print("create weight save file")
        except:
            print("already exist weight save file")

    with open(os.path.join(args.log_path,"config.txt"),'w') as f:
            str1 = 'name: ' +  str(args.name) + '\n'
            f.write(str1)
            str1 = 'opt: ' +  str(args.opt_type) + '\n'
            f.write(str1)
            str1 = 'lr: ' +  str(args.lr) + '\n'
            f.write(str1)
            str1 = 'w_decay: ' +  str(args.w_decay) + '\n'
            f.write(str1)
            str1 = 'beta1: ' +  str(args.beta1) + '\n'
            f.write(str1)
            str1 = 'beta2: ' +  str(args.beta2) + '\n'
            f.write(str1)
            str1 = 'nesterov: ' +  str(args.nesterov) + '\n'
            f.write(str1)
            str1 = 'auto_lr_tpye: ' +  str(args.lr_tpye) + '\n'
            f.write(str1)
            str1 = 'patience: ' +  str(args.patience) + '\n'
            f.write(str1)
            str1 = 'factor: ' +  str(args.factor) + '\n'
            f.write(str1)
            str1 = 'batch size: ' +  str(batch_size) + '\n'
            f.write(str1)
            str1 = 'step: ' +  str(args.step[0]) +" "+  str(args.step[1]) + '\n'
            f.write(str1)
            str1 = 'loader: ' +  str(args.loader) + '\n'
            f.write(str1)
            str1 = 'net: ' +  str(args.net) + '\n'
            f.write(str1)
            str1 = 'loss: ' +  str(args.loss) + '\n'
            f.write(str1)
            str1 = 'multi_lr: ' +  str(args.multi_lr) + '\n'
            f.write(str1)
            str1 = 'bias_decay: ' +  str(args.bias_decay) + '\n'
            f.write(str1)
            str1 = 'pre_: ' +  str(args.pre_) + '\n'
            f.write(str1)



    
def load_checkpoints(model,optimizer,lr_scheduler,args):
    """
    load checkpoints for models in the following order
    1. load old checkpoints
    2. load the optimizer, lr_scheduler, model_weight and epoch
    3. load imgnet per train model if no checkpoints
    """
    try:
        checkpoint = torch.load(args.checkpoint)
        model_state = checkpoint['model_state']
        opt_state = checkpoint['opt_state']
        lr_state = checkpoint['lr_state']
        start_epoch = checkpoint['epoch']
        print("load checkpoint success")
        try:
            model.load_state_dict(model_state)
        except:
            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        print("init network success") 
        try:
            optimizer.load_state_dict(opt_state)
            print('load opt state success')
        except:
            print('load opt state failed')
        try:
            lr_scheduler.load_state_dict(lr_state)
            print('load lr state success')
        except:
            print('load lr state failed')   

    except:
        start_epoch = 0
        print("no checkpoints to load")
        model_state = model_zoo.load_url(args.weight_vgg19, model_dir=args.weight_pre)
        vgg_keys = model_state.keys()
        pretrain_state = {}
        for i in range(20):
            pretrain_state[list(model.state_dict().keys())[i]
                    ] = model_state[list(vgg_keys)[i]]
        model_state = model.state_dict()
        model_state.update(pretrain_state)  
        print("load imgnet pretrain weight")
        model.load_state_dict(model_state)
        print("init network success")

    return start_epoch


def save_checkpoints(model,optimizer,lr_scheduler,epoch,val_loss,val_min,args):
    """
    save the min val loss and every train loss
    """
    train_path = os.path.join(args.weight_path,'train_final.pth')
    states = { 
               'model_state': model.state_dict(),
               'epoch': epoch + 1,
               'opt_state': optimizer.state_dict(),
               'lr_state': lr_scheduler.state_dict(),
    }
    torch.save(states,train_path)
    if val_loss<val_min:
        val_path = os.path.join(args.weight_path,'val_final.pth')
        torch.save(states,val_path)
        val_min = val_loss
    
    return val_min


def optimizer_settings(freeze_or_not,model,args):
    """
    1. choose different optimizer method here 
    2. default is SGD with momentum
    """
    if freeze_or_not:
        for i in range(20):
            for param in model.module.model0[i].parameters():
                param.requires_grad = False
        trainable_vars = [param for param in model.parameters() if param.requires_grad]
        if args.opt_type == 'sgd':
            optimizer = torch.optim.SGD(trainable_vars,
                                    lr = args.lr,
                                    momentum = args.beta1,
                                    weight_decay = args.w_decay,
                                    nesterov = args.nesterov)
        elif args.opt_type == 'adam':
            optimizer = torch.optim.Adam(trainable_vars, 
                                        lr=args.lr, 
                                        betas=(args.beta1, 0.999),
                                        eps=1e-08, 
                                        weight_decay=args.w_decay,
                                        amsgrad=False)
        else: print('opt type error, please choose sgd or adam')

    else:
        for param in model.module.parameters():
            param.requires_grad = True
        decay_1, decay_4, no_decay_2, no_decay_8 = [],[],[],[]
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print("some param freezed") 
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                if name[7:14] == "block_0" or name[7:14] == "block_1":
                    #print(name[7:14])
                    no_decay_2.append(param)
                    
                else:
                    #print(name[7:14])
                    no_decay_8.append(param)
            else:
                if name[7:14] == "block_0" or name[7:14] == "block_1":
                    decay_1.append(param)
                    
                else:
                    #print(name[7:14])
                    decay_4.append(param)
        
        if args.opt_type == 'sgd':
            optimizer = torch.optim.SGD([{'params': decay_1},
                                    {'params': decay_4,'lr': args.lr*4},
                                    {'params': no_decay_2,'lr': args.lr*2, 'weight_decay':0. },
                                    {'params': no_decay_8,'lr': args.lr*8,'weight_decay':0.}],
                                    lr = args.lr,
                                    momentum = args.beta1,
                                    weight_decay = args.w_decay,
                                    nesterov = args.nesterov)
        elif args.opt_type == 'adam':
            optimizer = torch.optim.Adam(trainable_vars, 
                                        lr=args.lr, 
                                        betas=(args.beta1, 0.999),
                                        eps=1e-08, 
                                        weight_decay=args.w_decay,
                                        amsgrad=False)
        else: print('opt type error, please choose sgd or adam')
    
    if args.lr_tpye == 'v_au':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, 
                                    verbose=True, threshold=1e-4, threshold_mode='rel',
                                    cooldown=3, min_lr=0, eps=1e-08)
    elif args.lr_tpye == 'ms': 
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step, gamma=args.factor, last_epoch=-1)
    
    else: print('lr_scheduler type error, please choose ms or v_au')
    
    return optimizer,lr_scheduler


def train_one_epoch(img_input,model,optimizer,writer,epoch,args):
    """
    Finish 
    1.train for one epoch
    2.print process, total loss, data time in terminal
    3.save loss, lr, output img in tensorboard
    Note   
    1.you can change the save frequency 
    """
    loss_train = 0
    model.train()
    length = len(img_input)
    print("iteration:",length)
    train_time = time.time()
    begin = time.time()
    
    '''loss control'''
    loss_for_control = torch.zeros([6,args.paf_num+args.heatmap_num])
    weight_con = torch.ones([1,args.paf_num+args.heatmap_num])
    weight_con = weight_con.cuda()
    
    '''start training'''
    for each_batch, (img, target_heatmap, heat_mask, target_paf, paf_mask) in enumerate(img_input):
        data_time = time.time() - begin

        img = img.cuda()
        target_heatmap = target_heatmap.cuda()
        target_paf = target_paf.cuda()
        heat_mask = heat_mask.cuda()
        paf_mask = paf_mask.cuda()
    
        _, saved_for_loss = model(img)
        #loss = CMUnet_loss.get_loss(saved_for_loss,target_heatmap,target_paf,args,weight_con)
        loss = CMUnet_loss.get_mask_loss(saved_for_loss,target_heatmap,heat_mask,target_paf,paf_mask,args,weight_con)

        # for i in range(args.paf_stage):
        #     for j in range(args.paf_num):
        #         loss_for_control[i][j] += loss['stage_{0}_{1}'.format(i,j)]
        # for i in range(len(saved_for_loss)-args.paf_stage):
        #     for j in range(args.heatmap_num):
        #         loss_for_control[i][j] += loss['stage_{0}_{1}'.format(i,j)]

        optimizer.zero_grad()
        loss["final"].backward()
        optimizer.step()
        loss_train += loss["final"]
    
        if each_batch % args.print_fre == 0:
            print_to_terminal_old(epoch,each_batch,length,loss,loss_train,data_time)
            #print_to_terminal(epoch,each_batch,length,loss,loss_train,data_time)
            #writer.add_scalar("train_loss_iterations", loss_train, each_batch + epoch * length)   
        begin = time.time()

        '''for short test'''
        # if each_batch == 5:
        #     break
    #weight_con = Online_weight_control(loss_for_control)
    loss_train /= length
    train_time = time.time() - train_time
    print('total training time:',train_time)
    return loss_train


def print_to_terminal(epoch,current_step,len_of_input,loss,loss_avg,datatime):
    """
    some public print information for both train and val
    """    
    str_print = "Epoch: [{0}][{1}/{2}\t]".format(epoch,current_step,len_of_input)
    str_print += "Total_loss: {loss:.4f}({loss_avg:.4f})".format(loss = loss['final'],
                            loss_avg = loss_avg/(current_step+1))
    str_print += "loss0: {loss:.4f}  ".format(loss = loss['stage_0'])
    str_print += "loss1: {loss:.4f}  ".format(loss = loss['stage_1'])
    str_print += "loss2: {loss:.4f}  ".format(loss = loss['stage_2'])
    str_print += "loss3: {loss:.4f}  ".format(loss = loss['stage_3'])
    str_print += "loss4: {loss:.4f}  ".format(loss = loss['stage_4'])
    str_print += "loss5: {loss:.4f}  ".format(loss = loss['stage_5'])
    str_print += "data_time: {time:.3f}".format(time = datatime)
    print(str_print)


def print_to_terminal_old(epoch,current_step,len_of_input,loss,loss_avg,datatime):
    """
    some public print information for both train and val
    """    
    str_print = "Epoch: [{0}][{1}/{2}\t]".format(epoch,current_step,len_of_input)
    str_print += "Total_loss: {loss:.4f}({loss_avg:.4f})".format(loss = loss['final'],
                            loss_avg = loss_avg/(current_step+1))
    str_print += "loss1_0: {loss:.4f}  ".format(loss = loss['stage_1_0'])
    str_print += "loss1_1: {loss:.4f}  ".format(loss = loss['stage_1_1'])
    str_print += "loss1_5: {loss:.4f}  ".format(loss = loss['stage_1_5'])
    str_print += "loss2_0: {loss:.4f}  ".format(loss = loss['stage_2_0'])
    str_print += "loss2_1: {loss:.4f}  ".format(loss = loss['stage_2_1'])
    str_print += "loss2_5: {loss:.4f}  ".format(loss = loss['stage_2_5'])
    str_print += "data_time: {time:.3f}".format(time = datatime)
    print(str_print)


def val_one_epoch(img_input,model,epoch,args):
    """ 
    val_type: 
    0.only calculate val_loss
    1.only calculate accuracy
    2.both accuracy and val_loss
    Note:     
    1.accuracy is single scale
    2.for multi-scale acc, run evaluate.py
    """
    loss_val, accuracy = 0,0
    json_output = []
    model.eval()
    length = len(img_input)
    begin = time.time()
    val_begin = time.time()

    # temporary
    weight_con = torch.ones([1,args.paf_num+args.heatmap_num])
    weight_con = weight_con.cuda()
    
    with torch.no_grad():
        for  each_batch, (img, target_heatmap, heat_mask, target_paf, paf_mask) in enumerate(img_input):
            # if each_batch == 5:
            #     break
            data_time = time.time() - begin
            img = img.cuda()
            target_heatmap = target_heatmap.cuda()
            target_paf = target_paf.cuda()
            heat_mask = heat_mask.cuda()
            paf_mask = paf_mask.cuda()

            if args.val_type == 0:
                _, saved_for_loss = model(img)
                loss = CMUnet_loss.get_mask_loss(saved_for_loss,target_heatmap,heat_mask,target_paf,paf_mask,args,weight_con)
                loss_val += loss['final']
        
            
            if each_batch % args.print_fre == 0:
                print_to_terminal_old(epoch,each_batch,length,loss,loss_val,data_time)
                #print_to_terminal(epoch,each_batch,length,loss,loss_val,data_time)
            begin = time.time()
        loss_val /= len(img_input)        
        #     elif args.val_type == 1:
        #         output, saved_for_loss = model(img)
        #         json_output = Callfromtrain(output,json_output)
        #         loss['final'] = 0
        #     else:
        #         output, saved_for_loss = model(img)
        #         loss = CMUnet_loss.get_loss(saved_for_loss,target_heatmap,target_paf,args,weight_con)
        #         accuracy = Callfromtrain(output,json_output)

        #     if each_batch % args.print_fre == 0:
        #         print_to_terminal(epoch,each_batch,length,loss,loss_val,data_time)
        #     begin = time.time()
        #     loss_val += loss['final']
        # loss_val /= len(img_input)
        # if args.val_type != 0:
        #     json_path = os.path.join(args.result_json,'_{}'.format(epoch),".json") 
        #     with open(args.result_json, 'w') as f:
        #         json.dump(json_output, f)
        #     evaluate.eval_coco(outputs=json_output, json_=json_path, ann_=args.ann_path)

    val_time = time.time() - val_begin
    print('total val time:',val_time)
    return loss_val, accuracy


def Online_weight_control(loss_list,args):
    """
    """
    loss_paf_ = torch.zeros([args.paf_num])
    loss_heat_ = torch.zeros([args.heatmap_num])
    for i in range(args.paf_stage):
        for j in range(args.paf_num):
            loss_paf_[j] += loss_list[i][j]
    for i in range(6-args.paf_stage):
        for j in range(args.heatmap_num):
            loss_heat_[j] += loss_list[i][j]
    print('losspaf',loss_paf_)
    print('lossheat',loss_heat_)
    ratio_paf = torch.min(loss_paf_)
    ratio_heat = torch.min(loss_heat_)
    loss_paf_ /= ratio_paf
    loss_heat_ /= ratio_heat
    print('losspaf_after',loss_paf_)
    print('lossheat_after',loss_heat_)
    weight_con = torch.cat([loss_paf_,loss_heat_],0)
    print('weicon',weight_con)
    
    return weight_con


if __name__ == "__main__":
    main()
    