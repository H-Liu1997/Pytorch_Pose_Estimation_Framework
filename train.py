# The train portion of total framework 
# __author__ = "Haiyang Liu"

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

from .datasets import mainloader
from .network.openpose import CMUnet, CMUnet_loss
from . import evaluate

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    CMUnet.cli(parser)
    CMUnet_loss.cli(parser)
    mainloader.train_cli(parser)
    evaluate.val_cli(parser)

    parser.add_argument('--stride_apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of epochs to train')
    parser.add_argument('--freeze_base', default=0, type=int,
                        help='number of epochs to train with frozen base')
    parser.add_argument('--square_edge', default=368, type=int,
                        help='square edge of input images')
    parser.add_argument('--no_augmentation', dest='augmentation',
                        default=True, action='store_false',
                        help='do not apply data augmentation')
    parser.add_argument('--log_path', default='./Pytorch_Pose_Estimation_Framework/ForSave/log/openpose1')
    parser.add_argument('--print_fre', default=10, type=int)
    parser.add_argument('--val_type', default=0, type=int)

    
    args = parser.parse_args()
    return args

def Online_weight_control(loss_dict):
    pass
    return loss_dict

def train_one_epoch(img_input,model,optimizer,writer,epoch,args):
    ''' Finish 1.train for one epoch
               2.print process, total loss, data time in terminal
               3.save loss, lr, output img in tensorboard
        Note   1.you can change the save frequency in config file
    '''
    loss_train = 0
    loss_for_control = torch.zeros([6,args.paf_num+args.heatmap_num])
    weight_con = torch.ones([1,args.paf_num+args.heatmap_num])
    weight_con = weight_con.cuda()
    model.train()
    length = len(img_input)
    
    begin = time.time()
    for each_batch, (img, target_heatmap, target_paf) in enumerate(img_input):
        data_time = time.time() - begin

        img = img.cuda()
        target_heatmap = target_heatmap.cuda()
        target_paf = target_paf.cuda()
    
        _, saved_for_loss = model(img)
        loss = CMUnet_loss.get_loss(saved_for_loss,target_heatmap,target_paf,args,weight_con)

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
            #for tensorboard
            print_to_terminal(epoch,each_batch,length,loss,loss_train,data_time)
            #writer.add_scalars()
            #writer.add_hyper()
            #writer.add_img()    
        begin = time.time()

        # if each_batch == 5:
        #     break
    #weight_con = Online_weight_control(loss_for_control)
    loss_train /= length
    return loss_train

def print_to_terminal(epoch,current_step,len_of_input,loss,loss_avg,datatime):
    ''' some public print information for both train and val
    '''    
    str_print = "Epoch: [{0}][{1}/{2}\t]".format(epoch,current_step,len_of_input)
    str_print += "Total_loss: {loss:.4f}({loss_avg:.4f})".format(loss = loss['final'],
                            loss_avg = loss_avg/(current_step+1))
    str_print += "loss0: {loss:.4f}".format(loss = loss['stage_0'])
    str_print += "loss0: {loss:.4f}".format(loss = loss['stage_1'])
    str_print += "loss0: {loss:.4f}".format(loss = loss['stage_2'])
    str_print += "loss0: {loss:.4f}".format(loss = loss['stage_3'])
    str_print += "loss0: {loss:.4f}".format(loss = loss['stage_4'])
    str_print += "loss0: {loss:.4f}".format(loss = loss['stage_5'])
    str_print += "data_time: {time:.3f}".format(time = datatime)
    print(str_print)

def val_one_epoch(img_input,model,epoch,args):
    ''' val_type: 0.only calculate val_loss
                  1.only calculate accuracy
                  2.both accuracy and val_loss
        Note:     1.accuracy is single scale
                  2.for multi-scale acc, run evaluate.py
    '''
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
        for each_batch, (img, target_heatmap, target_paf) in enumerate(img_input):
            data_time = time.time() - begin
            img = img.cuda()
            target_heatmap = target_heatmap.cuda()
            target_paf = target_paf.cuda()

            if args.val_type == 0:
                _, saved_for_loss = model(img)
                loss = CMUnet_loss.get_loss(saved_for_loss,target_heatmap,target_paf,args,weight_con)
                
            elif args.val_type == 1:
                output, saved_for_loss = model(img)
                json_output = Callfromtrain(output,json_output)
                loss['final'] = 0
            else:
                output, saved_for_loss = model(img)
                loss = CMUnet_loss.get_loss(saved_for_loss,target_heatmap,target_paf,args,weight_con)
                accuracy = Callfromtrain(output,json_output)

            if each_batch % args.print_fre == 0:
                print_to_terminal(epoch,each_batch,length,loss,loss_val,data_time)
            begin = time.time()
            loss_val += loss['final']
        loss_val /= len(img_input)
        if args.val_type != 0:
            json_path = os.join(args.result_json,'_{}'.format(epoch),".json") 
            with open(args.result_json, 'w') as f:
                json.dump(json_output, f)
            evaluate.eval_coco(outputs=json_output, json_=json_path, ann_=args.ann_path)

    val_time = time.time() - val_begin
    print('total val time:',val_time)
    return loss_val, accuracy

def optimizer_settings(freeze_or_not,model,args):
    ''' choose different optimizer method here 
    
        default is SGD and don't use nesv
    '''
    if freeze_or_not:
        for i in range(20):
            for param in model.module.model0[i].parameters():
                param.requires_grad = False
        trainable_vars = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(trainable_vars,
                                lr = args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay,
                                nesterov = args.nesterov)
    else:
        for param in model.module.parameters():
            param.requires_grad = True
        trainable_vars = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(trainable_vars, lr=args.lr, betas=(args.momentum, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
        # optimizer = torch.optim.SGD(trainable_vars,
        #                         lr = args.lr,
        #                         momentum = args.momentum,
        #                         weight_decay = args.weight_decay,
        #                         nesterov = args.nesterov)        
    return optimizer


if __name__ == "__main__":
    #load config parameters
    args = cli()
    
    # data portion
    train_loader = mainloader.train_factory('train',args)
    val_loader = mainloader.train_factory('val',args)

    # network portion
    model = CMUnet.CMUnetwork(args)
    try:
        state_dict = torch.load(args.weight_load_dir)
        print("load old weight")
    except:
        state_dict = model_zoo.load_url(args.weight_vgg19, model_dir=args.weight_load_dir)
        vgg_keys = state_dict.keys()
        weight_load_dir = {}
        for i in range(20):
            weight_load_dir[list(model.state_dict().keys())[i]
                     ] = state_dict[list(vgg_keys)[i]]
        state_dict = model.state_dict()
        state_dict.update(weight_load_dir)  
        print("load imgnet pretrain weight")
    # add some debug in future

    try: 
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(model).cuda()
    print("init network success")

    # start training
    val_loss_min = np.inf
    writer = SummaryWriter(args.log_path)
    
    
    if args.freeze_base != 0:
        print("start freeze some weight training for epoch 0-{}".format(args.freeze_base)) 
        optimizer = optimizer_settings(True,model,args)

        for epoch in range(args.freeze_base):
            loss_train = train_one_epoch(train_loader,model,optimizer,writer,epoch,args)
            loss_val, accuracy_val = val_one_epoch(val_loader,model,epoch,args)
            # save to tensorboard
            writer.add_scalars('train_val_loss', {'train loss': loss_train,
                                                  'val loss': loss_val}, epoch)
            writer.add_scalar('accuracy', accuracy_val, epoch)

            # val_weight is best val_loss weights
            # save train_weight is for continue training
            if val_loss_min > loss_val:
                val_loss_min = min(val_loss_min,loss_val)
                torch.save(model.state_dict(),args.weight_save_val_dir)
            torch.save(model.state_dict(),args.weight_save_train_dir)
    

    print("start normal training") 
    optimizer = optimizer_settings(False,model,args)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, 
                                  verbose=True, threshold=0.0001, threshold_mode='rel',
                                  cooldown=3, min_lr=0, eps=1e-08)

    for epoch in range(args.epochs):
        loss_train = train_one_epoch(train_loader,model,optimizer,writer,epoch,args)
        loss_val, accuracy_val = val_one_epoch(val_loader,model,epoch,args)
        # save to tensorboard
        writer.add_scalars('train_val_loss', {'train loss': loss_train,
                                                'val loss': loss_val}, epoch)
        writer.add_scalar('accuracy', accuracy_val, epoch)

        # val_weight is best val_loss weights
        # save train_weight is for continue training
        if val_loss_min > loss_val:
            val_loss_min = min(val_loss_min,loss_val)
            torch.save(model.state_dict(),args.weight_save_val_dir)

        torch.save(model.state_dict(),args.weight_save_train_dir)

    writer.close()

    
