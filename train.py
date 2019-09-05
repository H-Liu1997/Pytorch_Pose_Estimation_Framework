# The train portion of total framework 
# __author__ = "Haiyang Liu"

import time
from configparser import ConfigParser
from collections import OrderedDict

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tensorboardX import SummaryWriter

from .data import (DatasetsLoader,ImgPreprocessing)
from .network.openpose import (CMUnet_loss,CMUnet)
from .encoder.OPencoder import Get_OP_GT 
from . import evaluate 

 
def train_val_one_epoch(epoch,img_input,model,gt_img,optimizer,writer):
    ''' finish 1.train for one epoch
               2.val for one epoch
               3.optional: calculate acc for one epoch
        input  img_input 
               model
    '''
    accuracy_val, loss_train, loss_val = 0,0,0
    model.train()
    
    begin = time.time()
    for mini_batch, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(img_input):
        data_time = time.time() - begin

        output = model(img)
        loss = CMUnet_loss(output,gt_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss['final']

        if mini_batch % config['print']['frequency'] == 0:
            #for terminal print
            str_print = "Epoch: [{0}][{1}/{2}\t]".format(epoch,mini_batch,len(img_input))
            str_print += "Total_loss: {loss:.4f}({loss_avg:.4f})".format(loss = loss,
                                    loss_avg = loss_train/(mini_batch+1))
            str_print += "data_time: {time:.3f}".format(time = data_time)
            print(str_print)
            #for tensorboard
            writer.add_scalars()
            writer.add_hyper()
            writer.add_img()    
        begin = time.time()

    loss_train /= length


    model.eval()
    for mini_batch, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(img_input):
        output = model()
        loss = CMUnet_loss(output,gt_img)
        undatemodel(loss)
        loss_val += loss['final']
    loss_val /= length
    
    for mini_batch in range(val):
        accuracy_val = evaluate(val)

    return accuracy_val, loss_train, loss_val

def train():
def val():
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
    pre_function = ImgPreprocessing(config['imgpreprocessing'])
    gt_function = Get_OP_GT(config['encoding'])
    train_img, val_img = DatasetsLoader.train_factory(config['dataloader'],pre_function,gt_function)

    # network portion
    model = CMUnet()
    try:
        state_dict = torch.load(config['weight']['load'])
        print("load old weight")
    except:
        state_dict = torch.load_url()
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
            accuracy_val,loss_train,loss_val = train_val_one_epoch(epoch,input_img,model,gt_img,
                                                            optimizer)
            # save to tensorboard
            writer.add_scalars('train_val_loss', {'train loss': loss_train,
                                                'val loss': loss_val}, epoch)
            writer.add_scalar('accuracy', accuracy_val, epoch)

            # val is best val_loss weights
            # save train is for continue training
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
        accuracy_val,loss_train,loss_val = train_one_epoch(input_img,model,gt_img)

        writer.add_scalars('train_val_loss', {'train loss': loss_train,
                                             'val loss': loss_val}, epoch)
        writer.add_scalar('accuracy', accuracy_val, epoch)

        lr_scheduler.step(loss_val)
        if val_loss_min > loss_val:
            val_loss_min = min(val_loss_min,loss_val)
            torch.save(model.state_dict(),config['weight']['val'])
        torch.save(model.state_dict(),config['weight']['train'])

    writer.close()

    
        




