# ------------------------------------------------------------------------------
# The train code of total framework 
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

import time
import argparse
from collections import OrderedDict
import json
import os
import logging

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.model_zoo as model_zoo
import numpy as np
from tensorboardX import SummaryWriter

from .datasets import loader_factory 
from .network import loss_factory
from .network import network_factory
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # This portion just for recording in txt file, the following name portion also need be changed
    parser.add_argument('--name',           default='op_new_focus2',       type=str)
    parser.add_argument('--net_name',       default='CMU_new',          type=str)
    parser.add_argument('--loss',           default='focus_mask',     type=str)
    parser.add_argument('--loader',         default='CMU_117K',         type=str)
    network_factory.net_cli(parser,'CMU_new')
    loss_factory.loss_cli(parser,'focus_mask')
    loader_factory.loader_cli(parser,"CMU_117K")
    evaluate.val_cli(parser)

    # TODO: Add warm up
    #       Try vgg process
    #       Add val in running time
    #       Modify logging and print

    # short test and pretrain
    parser.add_argument('--short_test',     default=False,              type=bool)
    parser.add_argument('--pre_train',      default=0,                  type=int)
    parser.add_argument('--freeze_base',    default=0,                  type=int,       help='number of epochs to train with frozen base')
    parser.add_argument('--pretrain_lr',    default=1e-6,               type=float)
    parser.add_argument('--pre_w_decay',    default=5e-4,               type=float)
    parser.add_argument('--pre_iters',      default=10,                 type=int)
    
    # tricks
    parser.add_argument('--multi_lr',       default=True,               type=bool)
    parser.add_argument('--bias_decay',     default=True,               type=bool)
    parser.add_argument('--preprocess',     default='rtpose',           type=str)

    # other setting
    parser.add_argument('--seed',           default=7,                  type=int)
    parser.add_argument('--print_fre',      default=20,                 type=int)
    parser.add_argument('--val_type',       default=0,                  type=int)

    # trian setting
    parser.add_argument('--epochs',         default=300,                type=int)
    parser.add_argument('--per_batch',      default=10,                 type=int,       help='batch size per gpu')
    parser.add_argument('--gpu',            default=[0],                type=list,      help="gpu number")
    
    # optimizer
    parser.add_argument('--opt_type',       default='adam',             type=str,       help='sgd or adam')
    parser.add_argument('--lr',             default=1e-4,               type=float)
    parser.add_argument('--w_decay',        default=5e-4,               type=float)
    parser.add_argument('--beta1',          default=0.90,               type=float)
    parser.add_argument('--beta2',          default=0.999,              type=float)
    parser.add_argument('--nesterov',       default=True,               type=bool,      help='for sgd')

    parser.add_argument('--lr_tpye',        default='ms',               type=str,       help='milestone or auto_val')
    parser.add_argument('--factor',         default=0.5,                type=float,     help='divide factor of lr')
    parser.add_argument('--patience',       default=5,                  type=int)
    parser.add_argument('--step',           default=[200000,300000,360000,420000,480000,
                                                     540000,600000,700000,800000,900000],      type=list)
    #other path
    parser.add_argument('--log_base',       default="./Pytorch_Pose_Estimation_Framework/ForSave/log/")
    parser.add_argument('--weight_pre',     default="./Pytorch_Pose_Estimation_Framework/ForSave/weight/pretrain/")
    parser.add_argument('--weight_base',    default="./Pytorch_Pose_Estimation_Framework/ForSave/weight/")
    parser.add_argument('--checkpoint',     default="./Pytorch_Pose_Estimation_Framework/ForSave/weight/op_new_focus2/train_final.pth")
    
    args = parser.parse_args()
    return args


def main():
    '''load config parameters'''
    args = cli()
    save_config(args)
    
    '''deterministic'''
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    '''data portion'''
    train_factory = loader_factory.loader_factory(args)
    train_loader = train_factory('train',args)
    val_loader = train_factory('val',args)
    
    '''network portion'''  
    model = network_factory.get_network(args)
    # multi_gpu and cuda, will occur some bug when inner some function
    model = torch.nn.DataParallel(model,args.gpu).cuda()
    optimizer,lr_scheduler = optimizer_settings(False,model,args)
    start_epoch = load_checkpoints(model,optimizer,lr_scheduler,args)
   
    '''val loss boundary and tensorboard path'''
    loss_function = loss_factory.get_loss_function(args)
    #TODO loss function using same parameters
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
            loss_train = pretrain_one_epoch(train_loader,model,optimizer,writer,epoch,args,loss_function)
            writer.add_scalar('train_pre_loss', loss_train, epoch)
            
    '''start normal training'''
    print("start normal training")
    if flag: 
        optimizer,lr_scheduler = optimizer_settings(False,model,args)
        start_epoch = args.freeze_base
    for epoch in range(start_epoch,args.epochs):
        loss_train = train_one_epoch(train_loader,model,optimizer,writer,epoch,args,loss_function,lr_scheduler)
        loss_val, accuracy_val = val_one_epoch(val_loader,model,epoch,args,loss_function)

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
    # modify some parameters
    batch_size = len(args.gpu) * args.per_batch
    args.log_path = os.path.join(args.log_base,args.name)
    args.weight_path = os.path.join(args.weight_base,args.name)
    args.batch_size = batch_size
    flag_have_file = 0
    
    # create config save file
    try:
        os.mkdir(args.log_path)
        logging.basicConfig(filename=os.path.join(args.log_base,(args.name+'.log')),
            format='%(levelname)s:%(message)s', level=logging.INFO)
        print("create log save file")       
    except:
        flag_have_file = 1
        logging.basicConfig(filename=os.path.join(args.log_base,(args.name+'.log')),
            format='%(levelname)s:%(message)s', level=logging.INFO)
        print('already exist the log file, please remove them if needed')
    try:
        os.mkdir(args.weight_path)
        print("create weight save file")
    except:
        print("already exist weight save file, please remove them if needed")

    #write log information
    if flag_have_file==1:
        logging.info('-----------------Continue-----------------')
        logging.info('Continue Seed: %s',str(args.seed))
    else:
        logging.info('------------------Start-----------------')
        logging.info('Experimental Name: %s',   args.name)
        logging.info('----------------Optimizer-Info----------------')
        logging.info('Optimizer: %s',           args.opt_type)
        logging.info('Learning Rate: %s',       str(args.lr))
        logging.info('Weight Decay: %s',        str(args.w_decay))
        logging.info('Beta1 or Momentum: %s',   str(args.beta1))

        if args.opt_type == 'sgd':
            logging.info('SGD nesterov: %s',        str(args.nesterov))
        else:
            logging.info('Beta2: %s',               str(args.beta2))

        logging.info('Auto_lr_tpye: %s',        str(args.lr_tpye))
        if args.lr_tpye == 'ms':
            logging.info('Factor: %s',        str(args.factor))
            logging.info('Step: %s',          str(args.step))
        else:
            logging.info('Patience: %s',      str(args.patience))
        
        logging.info('----------------Train-Info----------------')
        logging.info('GPU: %s',                 str(args.gpu))
        logging.info('Batch Szie Total: %s',    str(batch_size))
        logging.info('Batch Szie: %s',          str(batch_size))
        logging.info('No Bias Decay: %s',       str(args.bias_decay))
        logging.info('Multi Lr: %s',            str(args.multi_lr))

        logging.info('----------------Data-Info----------------')
        logging.info('Data Type: %s',           str(args.loader))
        logging.info('Preprocess Type: %s',     str(args.preprocess))
        logging.info('Scale shown in the name')
        
        logging.info('----------------Other-Info----------------')
        logging.info('Network Tpye: %s', args.net_name)
        logging.info('Loss Type: %s', args.loss)
        logging.info('Start Seed: %s', str(args.seed))


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
        logging.info('Epoch: %s', str(start_epoch))
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
            print('load lr state success','lr: ',optimizer.param_groups[0]['lr'])
        except:
            print('load lr state failed','lr: ',optimizer.param_groups[0]['lr'])   

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
        try:
            for param in model.module.block_0.parameters():
                param.requires_grad = False
            for param in model.module.block_0.conv4_3.parameters():
                param.requires_grad = True
            for param in model.module.block_0.conv4_4.parameters():
                param.requires_grad = True
        except:
            print("error! freeze need change base on network")

        trainable_vars = [param for param in model.parameters() if param.requires_grad]
        if args.opt_type == 'sgd':
            optimizer = torch.optim.SGD(trainable_vars,
                                    lr = args.pretrain_lr,
                                    momentum = args.beta1,
                                    weight_decay = args.pre_w_decay,
                                    nesterov = args.nesterov)
        elif args.opt_type == 'adam':
            optimizer = torch.optim.Adam(trainable_vars, 
                                        lr=args.pretrain_lr, 
                                        betas=(args.beta1, 0.999),
                                        eps=1e-08, 
                                        weight_decay=args.pre_w_decay,
                                        amsgrad=False)
        else: print('opt type error, please choose sgd or adam')

    else:
        for param in model.module.parameters():
            param.requires_grad = True

        if args.multi_lr:
            decay_1, decay_4, no_decay_2, no_decay_8 = [],[],[],[]
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print("some param freezed") 
                    continue
                if args.net_name =='CMU_new':
                    if name.endswith(".bias"):
                        if name[7:14] == "state_0":
                            print(name[7:],"using no_decay_2")
                            no_decay_2.append(param)
                            
                        else:
                            no_decay_8.append(param)
                            print(name[7:],"using no_decay_8")
                    else:
                        if name[7:14] == "state_0":
                            decay_1.append(param)
                            print(name[7:],"using decay_1")
                            
                        else:
                            #print(name[7:14])
                            decay_4.append(param)
                            print(name[7:],"using decay_4")
                else:
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

            params = [  {'params': decay_1, 'lr': args.lr, 'weight_decay':args.w_decay,},
                        {'params': no_decay_2, 'lr': args.lr*2,'weight_decay':0,},
                        {'params': decay_4, 'lr': args.lr*4,'weight_decay':args.w_decay,},
                        {'params': no_decay_8, 'lr': args.lr*8,'weight_decay':0,}]
            
            if args.opt_type == 'sgd':
                optimizer = torch.optim.SGD(params,
                                        lr = args.lr,
                                        momentum = args.beta1,
                                        weight_decay = args.w_decay,
                                        nesterov = args.nesterov)
            elif args.opt_type == 'adam':
                optimizer = torch.optim.Adam(params,
                                            lr=args.lr, 
                                            betas=(args.beta1, 0.999),
                                            eps=1e-08, 
                                            weight_decay=args.w_decay,
                                            amsgrad=False)
            else: print('opt type error, please choose sgd or adam')
        else: 
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
    
    if args.lr_tpye == 'v_au':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, 
                                    verbose=True, threshold=1e-4, threshold_mode='rel',
                                    cooldown=3, min_lr=0, eps=1e-08)
    elif args.lr_tpye == 'ms': 
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step, gamma=args.factor, last_epoch=-1)
    
    else: 
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step, gamma=1, last_epoch=-1)
        print('lr_scheduler type error, please choose ms or v_au')
    
    return optimizer,lr_scheduler


def pretrain_one_epoch(img_input,model,optimizer,writer,epoch,args,loss_function):
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
    lr = 0
    
    '''loss control'''
    loss_for_control = torch.zeros([6,args.paf_num+args.heatmap_num])
    weight_con = torch.ones([1,args.paf_num+args.heatmap_num])
    weight_con = weight_con.cuda()
    
    '''start training'''
    for each_batch, (img, target_heatmap, heat_mask, target_paf, paf_mask) in enumerate(img_input):
        if each_batch == args.pre_iters:
            print("pretrain finish")
            break
        data_time = time.time() - begin
        img = img.cuda()
        target_heatmap = target_heatmap.cuda()
        target_paf = target_paf.cuda()
        heat_mask = heat_mask.cuda()
        paf_mask = paf_mask.cuda()
    
        _, saved_for_loss = model(img)
        #loss = CMUnet_loss.get_loss(saved_for_loss,target_heatmap,target_paf,args,weight_con)
        loss = loss_function(saved_for_loss,target_heatmap,heat_mask,target_paf,paf_mask,args,weight_con)

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
            if args.loss == 'CMU_new_mask':
                print_to_terminal(epoch,each_batch,length,loss,loss_train,data_time,lr)
            else:    
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


def train_one_epoch(img_input,model,optimizer,writer,epoch,args,loss_function,lr_scheduler):
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
        loss = loss_function(saved_for_loss,target_heatmap,heat_mask,target_paf,paf_mask,args,weight_con)

        # for i in range(args.paf_stage):
        #     for j in range(args.paf_num):
        #         loss_for_control[i][j] += loss['stage_{0}_{1}'.format(i,j)]
        # for i in range(len(saved_for_loss)-args.paf_stage):
        #     for j in range(args.heatmap_num):
        #         loss_for_control[i][j] += loss['stage_{0}_{1}'.format(i,j)]

        optimizer.zero_grad()
        loss["final"].backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        loss_train += loss["final"]
    
        if each_batch % args.print_fre == 0:
            if args.loss == 'CMU_new_mask' or args.loss == 'focus_mask':
                print_to_terminal(epoch,each_batch,length,loss,loss_train,data_time,lr)
            else:    
                print_to_terminal_old(epoch,each_batch,length,loss,loss_train,data_time)
            #writer.add_scalar("train_loss_iterations", loss_train, each_batch + epoch * length)   
        begin = time.time()

        '''for short test'''
        if args.short_test and each_batch == 5:
            break
        
    #weight_con = Online_weight_control(loss_for_control)
    loss_train /= length
    train_time = time.time() - train_time
    print('total training time:',train_time)
    return loss_train


def print_to_terminal(epoch,current_step,len_of_input,loss,loss_avg,datatime,lr):
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
    str_print += "lr: {lr:} ".format(lr = lr)

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


def val_one_epoch(img_input,model,epoch,args,loss_function):
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
    lr = 0
    # temporary
    weight_con = torch.ones([1,args.paf_num+args.heatmap_num])
    weight_con = weight_con.cuda()
    
    with torch.no_grad():
        for  each_batch, (img, target_heatmap, heat_mask, target_paf, paf_mask) in enumerate(img_input):
            if args.short_test and each_batch == 5:
                break
            data_time = time.time() - begin
            img = img.cuda()
            target_heatmap = target_heatmap.cuda()
            target_paf = target_paf.cuda()
            heat_mask = heat_mask.cuda()
            paf_mask = paf_mask.cuda()

            if args.val_type == 0:
                _, saved_for_loss = model(img)
                loss = loss_function(saved_for_loss,target_heatmap,heat_mask,target_paf,paf_mask,args,weight_con)
                loss_val += loss['final']
        
            
            if each_batch % args.print_fre == 0:
                if args.loss == 'CMU_new_mask' or args.loss == 'focus_mask':
                    print_to_terminal(epoch,each_batch,length,loss,loss_val,data_time,lr)
                else:    
                    print_to_terminal_old(epoch,each_batch,length,loss,loss_val,data_time)
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
    