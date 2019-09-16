# The train portion of total framework 
# __author__ = "Haiyang Liu"

import time
from configparser import ConfigParser
from collections import OrderedDict

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.model_zoo as model_zoo
import numpy as np
from tensorboardX import SummaryWriter

from .datasets import mainloader
from .network.openpose import CMUnet, CMUnet_loss
from .EvalTools import decoder, CalScore

def Online_weight_control(loss_dict):
    pass
    return

def train_one_epoch(img_input,model,optimizer,writer,epoch,loss_set):
    ''' Finish 1.train for one epoch
               2.print process, total loss, data time in terminal
               3.save loss, lr, output img in tensorboard
        Note   1.you can change the save frequency in config file
    '''
    loss_train = 0
    loss_for_control = np.zeros(6,19+38)
    weight_con = np.ones(1,19+38)
    weight_con = weight_con.cuda()
    model.train()
    length = len(img_input)
    
    begin = time.time()
    for each_batch, (img, target, mask) in enumerate(img_input):
        data_time = time.time() - begin

        img = img.cuda()
        target = target.cuda()
        mask = mask.cuda()

        _, saved_for_loss = model(img)
        loss_final,loss = CMUnet_loss.get_loss(saved_for_loss,target,mask,loss_set,weight_con)

        for i in range(6):
            for j in range(19+38):
                loss_for_control[i][j] += loss['stage_{0}_{1}'.format(i,j)]

        optimizer.zero_grad()
        loss_final.backward()
        optimizer.step()
        loss_train += loss_final

        if each_batch % config['print']['frequency'] == 0:
            #for tensorboard
            print_to_terminal(epoch,each_batch,length,loss_final,loss_train,data_time)
            writer.add_scalars()
            writer.add_hyper()
            writer.add_img()    
        begin = time.time()
    weight_con = Online_weight_control(loss_for_control)
    loss_train /= length
    return loss_train

def print_to_terminal(epoch,current_step,len_of_input,loss,loss_avg,datatime):
    ''' some public print information for both train and val
    '''    
    str_print = "Epoch: [{0}][{1}/{2}\t]".format(epoch,current_step,len_of_input)
    str_print += "Total_loss: {loss:.4f}({loss_avg:.4f})".format(loss = loss,
                            loss_avg = loss_avg/(current_step+1))
    str_print += "data_time: {time:.3f}".format(time = datatime)
    print(str_print)

def val_one_epoch(img_input,model,epoch,val_type,loss_set,decoder_set):
    ''' val_type: 0.only calculate val_loss
                  1.only calculate accuracy
                  2.both accuracy and val_loss
        Note:     1.accuracy is single scale
                  2.for multi-scale acc, run evaluate.py
    '''
    loss_val, accuracy = 0,0
    model.eval()
    length = len(img_input)
    begin = time.time()

    if val_type == 0:
        for each_batch, (img, target, mask) in enumerate(img_input):
            data_time = time.time() - begin
            img = img.cuda()
            target = target.cuda()
            mask = mask.cuda()

            _, saved_for_loss = model(img)
            loss = CMUnet_loss.get_loss(saved_for_loss,target,mask,loss_set)
            loss_val += loss
            if each_batch % config['print']['frequency'] == 0:
                print_to_terminal(epoch,each_batch,length,loss['final'],loss_val,data_time)
            begin = time.time()
        loss_val /= len(img_input)

    elif val_type == 1:
        for each_batch, (img, target, mask) in enumerate(img_input):
            data_time = time.time() - begin
            img = img.cuda()
            target = target.cuda()
            mask = mask.cuda()

            output, saved_for_loss = model(img)
            json_file += decoder(output,decoder_set)
            
            if each_batch % config['print']['frequency'] == 0:
                print_to_terminal(epoch,each_batch,length,0,0,data_time)
            begin = time.time()
        accuracy = CalScore(json_file)

    else:
        for each_batch, (img, target, mask) in enumerate(img_input):
            data_time = time.time() - begin
            img = img.cuda()
            target = target.cuda()
            mask = mask.cuda()

            output, saved_for_loss = model(img)
            json_file = decoder(output,decoder_set)
            

            loss = CMUnet_loss.get_loss(saved_for_loss,target,mask,loss_set)
            loss_val += loss
            if each_batch % config['print']['frequency'] == 0:
                print_to_terminal(epoch,each_batch,length,loss['final'],loss_val,data_time)
            begin = time.time() 
        loss_val /= len(img_input)
        accuracy = CalScore(json_file)

    val_time = time.time() - begin
    print('total val time:',val_time)
    return loss_val, accuracy

def optimizer_settings(freeze_or_not,model):
    ''' choose different optimizer method here 
    
        default is SGD and don't use nesv
    '''
    if freeze_or_not:
        for i in range(20):
            for param in model.module.model0[i].parameters():
                param.requires_grad = False
        trainable_vars = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(trainable_vars,
                                lr = config['train']['lr'],
                                momentum = config['train']['lr'],
                                weight_decay = config['train']['decay'],
                                nesterov = config['train']['nesterov'])
    else:
        for param in model.module.parameters():
            param.requires_grad = True
        trainable_vars = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(trainable_vars,
                                lr = config['train']['lr'],
                                momentum = config['train']['lr'],
                                weight_decay = config['train']['decay'],
                                nesterov = config['train']['nesterov'])         
    return optimizer


if __name__ == "__main__":
    config = ConfigParser()
    config.read("OP.config")
    print("Reading config file success")
    
    # data portion
    train_img, val_img = mainloader.train_factory(type_,args)

    # network portion
    model = CMUnet.CMUnetwork()
    try:
        state_dict = torch.load(config['weight']['load'])
        print("load old weight")
    except:
        state_dict = model_zoo.load_url(config['weight']['vgg19'], model_dir=config['weight']['load'])
        vgg_keys = state_dict.keys()
        weights_load = {}
        for i in range(20):
            weights_load[list(model.state_dict().keys())[i]
                     ] = state_dict[list(vgg_keys)[i]]
        state_dict = model.state_dict()
        state_dict.update(weights_load)  
        print("load imgnet pretrain weight")
    # add some debug in future
    model.load_state_dict(state_dict)
    model = torch.nn.Dataparallel(model).cuda()
    print("init network success")

    # start training
    val_loss_min = np.inf
    writer = SummaryWriter(config['log']['path'])
    
    
    if config['train']['freeze'] != 0:
        print("start freeze some weight training for epoch 0-{}".format(config['train']['freeze'])) 
        optimizer = optimizer_settings(True,model)

        for epoch in range(config['train']['freeze']):
            loss_train = train_one_epoch(train_img,model,optimizer,writer,epoch,config['loss_settings'])
            loss_val, accuracy_val = val_one_epoch(val_img,model,epoch,config['val']['type'],config['loss_settings'],config['decoder'])
            # save to tensorboard
            writer.add_scalars('train_val_loss', {'train loss': loss_train,
                                                  'val loss': loss_val}, epoch)
            writer.add_scalar('accuracy', accuracy_val, epoch)

            # val_weight is best val_loss weights
            # save train_weight is for continue training
            if val_loss_min > loss_val:
                val_loss_min = min(val_loss_min,loss_val)
                torch.save(model.state_dict(),config['weight']['val'])
            torch.save(model.state_dict(),config['weight']['train'])
    

    print("start normal training") 
    optimizer = optimizer_settings(False,model)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, 
                                  verbose=True, threshold=0.0001, threshold_mode='rel',
                                  cooldown=3, min_lr=0, eps=1e-08)

    for epoch in range(config['train']['freeze'],config['train']['epoch']):
        loss_train = train_one_epoch(train_img,model,optimizer,writer,epoch,config['loss_settings'])
        loss_val, accuracy_val = val_one_epoch(val_img,model,epoch,config['val']['type'],config['loss_settings'],config['decoder'])
        # save to tensorboard
        writer.add_scalars('train_val_loss', {'train loss': loss_train,
                                                'val loss': loss_val}, epoch)
        writer.add_scalar('accuracy', accuracy_val, epoch)

        # val_weight is best val_loss weights
        # save train_weight is for continue training
        if val_loss_min > loss_val:
            val_loss_min = min(val_loss_min,loss_val)
            torch.save(model.state_dict(),config['weight']['val'])
        torch.save(model.state_dict(),config['weight']['train'])

    writer.close()

    
