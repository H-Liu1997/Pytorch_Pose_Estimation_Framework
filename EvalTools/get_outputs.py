''' this file is for get multi_scale output, can test directly
    input Ori img
    output the network output(encoder) after average of Multi_scale(can set 1 scale)
    for the show img part, you should change the code base on your outputs
    by Haiyangliu 2019.8'''

import torch
import matplotlib.pyplot as plt # wrong 1 time
import numpy as np
import panda as pd
import cv2 # wrong 1 time 

from network import CPMmodel
from datasets import img_transform,img_preprocessing # wrong 1 time 
 
Scale = [0.5, 1, 1.5, 2, 2.5]
Filp_or_not = False
weight_path = 'path.pth'
GPU_id = [0,1]
heatmap_num = 18
paf_num = 43
img_val_path = 'path'
test_number = 1
pooling_factor = 8
base_size = 368 
preprocessing = 'vgg'


def Get_Multiple_outputs(input_img,model,Scale,Filp_or_not,heatmap_num,paf_num):
    ''' mini_batch for test implement by using multi_scale img for one batch '''
    average_heatmap = np.zeros((input_img.shape[0], input_img.shape[1], heatmap_num)) # wrong 3 time
    average_paf = np.zeros((input_img.shape[0], input_img.shape[1], paf_num)) # wrong 3 time

    multi_size = list( x * base_size for x in Scale)
    val_batch_numby,useful_shape = get_mini_batch(multi_size,input_img)

    val_batch = torch.from_numby(val_batch_numby).cuda().float() #wrong 1 time
    outputs, _ = model(val_batch)
    #forget some part
    outputs_paf, outputs_heatmap = outputs[-2], outputs[-1]
    pafs = outputs_paf.cpu().data.numpy().transpose(0, 2, 3, 1) # h/8 * w/8 * paf_number
    heatmaps = outputs_heatmap.cpu().data.numpy().transpose(0, 2, 3, 1)

    ''' return the size to ori size and get the average value ''' 
    for n_times in range(len(Scale)):
        ''' this part will occur some error because int()'''
        pafs_useful = pafs[n_times,:int(useful_shape[n_times][0] / 8),:int(useful_shape[n_times][1] / 8) ,:]
        heatmaps_useful = heatmaps[n_times,:int(useful_shape[n_times][0] / 8),:int(useful_shape[n_times][1] / 8),:]

        pafs_up = cv2.resize(pafs_useful, None, fx = pooling_factor, fy = pooling_factor, interpolation=cv2.INTER_CUBIC)
        heatmaps_up = cv2.resize(heatmaps_useful, None, fx = pooling_factor, fy = pooling_factor, interpolation=cv2.INTER_CUBIC)

        '''this part has some difference need check'''
        pafs_ori = cv2.resize(pafs_up,(input_img.shape[0],input_img.shape[1]),interpolation = cv2.INTER_CUBIC)
        heatmaps_ori = cv2.resize(heatmaps_up,(input_img.shape[0],input_img.shape[1]),interpolation = cv2.INTER_CUBIC)
        average_paf = average_paf + pafs_ori / len(Scale)
        average_heatmap = average_heatmap + heatmaps_ori / len(Scale)
        
    ''' if Filp = true, run the following code '''
    if Filp_or_not:
        pass
    return average_paf,average_heatmap


def data_loader_val(img_val_path):
    '''better to implement here, the data_loader_train may reuse the torchcv's data_loader
       information file format:
       1	136	COCO_val2014_000000000136.jpg	374	500
       2	192	COCO_val2014_000000000192.jpg	480	640
       3	241	COCO_val2014_000000000241.jpg	640	480 '''
    info_file = pd.read_csv(img_val_path, sep = '/s+', header = None)
    img_id = list(info_file[1])
    img_path = list(info_file[2])
    img_height = list(info_file[3])
    img_width = list(info_file[4])
    return img_id, img_path, img_height, img_width


def get_mini_batch(multi_size,input_img):
    ''' return scale_num * 3 * h * w numby array '''
    max_size = multi_size[-1]
    useful_shape = []
    
    ''' return the value module pooling_factor == 0 
        choose the min shape and change it to n * 368 '''
    max_input, _, _ = img_transform.crop_with_factor(input_img, max_size, 
                                                          factor = pooling_factor, is_ceil = True)
    val_batch_numby = np.zeros((len(multi_size), 3, max_input.shape[0], max_input.shape[1]))

    for i in range(len(multi_size)): # wrong 1 time
        input_crop, _, _ = img_transform.crop_with_factor(input_img, multi_size[i], 
                                                          factor = pooling_factor, is_ceil = True)
        '''change the input size from h * w * 3 to 3 * h * w and normalization for vgg '''                                                  
        if preprocessing == 'vgg':
            img_final = img_preprocessing.vgg_preprocess(input_crop)
        else:
            pass
        val_batch_numby[i, :, :img_final.shape[1], :img_final.shape[2]] = img_final # wrong 1 time
        useful_shape.append((img_final.shape[1],img_final.shape[2]))

    return val_batch_numby,useful_shape


if __name__ == "__main__":
    print("Scale number: ",len(Scale))
    print("Scale: ",Scale)
    print("Filp_or_not: ",Filp_or_not)
    print("weight: ",weight_path)
    print("start loading model")

    with torch.autograd.no_grad():
        state_dict = torch.load(weight_path)
        model = CPMmodel()
        model.load_state_dict(state_dict)
        model = torch.nn.DataParallel(model,GPU_id).cuda() # wrong 1 time
        model.eval()
        #model.float()
        #model = model.cuda() # wrong 1 time
    print("loading model sucess")

    img_id, img_path, img_height, img_width = data_loader_val(img_val_path)
    print("total val image number: ",len(img_id))

    ''' only test the test_number one image: h * w * 3'''
    img_one_time = cv2.imread(img_path[test_number]) 
    pafs, heatmaps = Get_Multiple_outputs(img_one_time,model,Scale,Filp_or_not,heatmap_num,paf_num)

    '''show and save the results'''
    #sometimes can learn better method
    plt.figure(num = 0, figsize = (5,8))
    plt.subplot(121)
    plt.imshow(pafs[0])
    plt.subplot(122)
    plt.imshow(heatmaps[0])
    plt.savefig("test_heatmap_paf.png") #wrong 1 time two reasons
    plt.show()
    