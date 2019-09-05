# The evaluate portion of pose estimation
# Some tools is from coco official
# __author__ = 'Haiyang Liu'

from configparser import ConfigParser

import torch

from .decoder import ToJson
from .decoder.ToKeypoints import (Get_kepoints,get_outputs)
from .data import (DataLoader,ImgPreprocessing,ImgTransform)
from .network.openpose import CMUnet

def CalculateScore():
    pass

def CallFromTrain(model,img_val,index):
    '''index 0: only val_loss
       index 1: only accuracy
       index 2: val_loss + accuracy
       The accuracy is single scale mAP
       for multi-scale please run evaluate.py
    '''
    outputs = get_outputs()
    if index == 0: 
        return outputs, 0
    else:
        outputs = get_outputs()
        data = Get_kepoints(outputs)
        json_file = ToJson(data)
        accuracy = CalculateScore(json_file)
        return outputs,accuracy

    




    pass
if __name__ == "__main__":
    config = ConfigParser()
    config.read("OP.config")
    print("read config sucess")

    img_val = DataLoader(config['dataloader_val'])
    img_val = ImgPreprocessing(config['imgpreprocessing'])

    model = CMUnet()
    state_dict = torch.load(config['weight']['eval'])
    model = model.load(state_dict)
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    for minibatch in range():









    

