# ------------------------------------------------------------------------------
# The evaluate portion of pose estimation 
# Some tools is from coco official
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

import json
from .EvalTools.decoder import post
from .EvalTools import eval_trans


def val_cli(parser):
    group = parser.add_argument_group('evaluate')
    group.add_argument('--result_img_dir',default="./Pytorch_Pose_Estimation_Framework/ForSave/imgs/openpose3",
                        type=str)
    group.add_argument('--result_json', default="./Pytorch_Pose_Estimation_Framework/ForSave/json/",
                        type=str)
    group.add_argument('--thre1', default=0.1,  type=float)
    group.add_argument('--thre2', default=0.05, type=float)
    group.add_argument('--thre3', default=0.5, type=float)
    group.add_argument('--pooling_factor', default=8, type=int)
    group.add_argument('--filp', default=False, type=bool) 


def CallFromTrain(model,img,target_heatmap,target_paf):
    '''index 0: only val_loss
       index 1: only accuracy
       index 2: val_loss + accuracy
       The accuracy is single scale mAP
       for multi-scale please run evaluate.py
    '''
    # outputs = get_outputs()
    # if index == 0: 
    #     return outputs, 0
    # else:
    #     outputs = get_outputs()
    #     data = Get_kepoints(outputs)
    #     json_file = ToJson(data)
    #     accuracy = CalculateScore(json_file)
    #     return outputs,accuracy
    pass

def get_mini_batch(multi_size,input_img):
    pooling_factor=8
    ''' return scale_num * 3 * h * w numby array '''
    useful_shape = []
    real_shape = []
    max_size = multi_size[-1]
    ''' return the value module pooling_factor == 0 
        choose the min shape and change it to n * 368 '''
    max_input, _, _ = eval_trans.crop_with_factor(input_img, max_size, 
                                                          factor = pooling_factor, is_ceil = True)
    val_batch_numpy = np.zeros((len(multi_size), 3, max_input.shape[0], max_input.shape[1]))

    for i in range(len(multi_size)): # wrong 1 time
        input_crop, _, real_shape_tem = eval_trans.crop_with_factor(input_img, multi_size[i], 
                                                          factor = pooling_factor, is_ceil = True)
        '''change the input size from h * w * 3 to 3 * h * w and normalization for vgg '''  
        img_final = eval_trans.rtpose_preprocess(input_crop) 
                                                  
        # if preprocessing == 'vgg':
        #     img_final = vgg_preprocess(input_crop)
        # else:
        #     pass
        val_batch_numpy[i, :, :img_final.shape[1], :img_final.shape[2]] = img_final # wrong 1 time
        useful_shape.append((img_final.shape[1],img_final.shape[2]))
        real_shape.append((real_shape_tem[0],real_shape_tem[1]))
    return val_batch_numpy,useful_shape,real_shape


def Get_Multiple_outputs(input_img,model,Scale,Filp_or_not,heatmap_num,paf_num,args):
    base_size=368
    pooling_factor=8
    ''' mini_batch for test implement by using multi_scale img for one batch '''
    average_heatmap = np.zeros((input_img.shape[0], input_img.shape[1], heatmap_num)) # wrong 3 time
    average_paf = np.zeros((input_img.shape[0], input_img.shape[1], paf_num)) # wrong 3 time

    multi_size = list( x * base_size for x in Scale)
    val_batch_numpy,useful_shape,real_shape = get_mini_batch(multi_size,input_img)

    val_batch = torch.from_numpy(val_batch_numpy).cuda().float() #wrong 1 time
    outputs, _ = model(val_batch)
    #outputs, _ = model(val_batch)
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

        '''this part has some difference need check
           the resize para(x,y) isn't height * width, is width * height '''
        
        pafs_up = pafs_up[0:real_shape[n_times][0], 0:real_shape[n_times][1], :]
        heatmaps_up = heatmaps_up[0:real_shape[n_times][0], 0:real_shape[n_times][1], :]
        pafs_ori = cv2.resize(pafs_up,(input_img.shape[1],input_img.shape[0]),interpolation = cv2.INTER_CUBIC)
        heatmaps_ori = cv2.resize(heatmaps_up,(input_img.shape[1],input_img.shape[0]),interpolation = cv2.INTER_CUBIC)
        average_paf = average_paf + pafs_ori / len(Scale)
        average_heatmap = average_heatmap + heatmaps_ori / len(Scale)
        
    ''' if Filp = true, run the following code '''
    if Filp_or_not:
        pass
    return average_paf,average_heatmap


def append_result(image_id, person_to_joint_assoc, joint_list, outputs):
    """Build the outputs to be evaluated
    :param image_id: int, the id of the current image
    :param person_to_joint_assoc: numpy array of joints associations
    :param joint_list: list, list of joints
    :param outputs: list of dictionaries with the following keys: image_id,
                    category_id, keypoints, score
    """
    ORDER_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    for ridxPred in range(len(person_to_joint_assoc)):
        one_result = {
            "image_id": 0,
            "category_id": 1,
            "keypoints": [],
            "score": 0
        }

        one_result["image_id"] = image_id
        keypoints = np.zeros((17, 3))

        for part in range(17):
            ind = ORDER_COCO[part]
            index = int(person_to_joint_assoc[ridxPred, ind])

            if -1 == index:
                keypoints[part, 0] = 0
                keypoints[part, 1] = 0
                keypoints[part, 2] = 0

            else:
                keypoints[part, 0] = joint_list[index, 0] + 0.5
                keypoints[part, 1] = joint_list[index, 1] + 0.5
                keypoints[part, 2] = 1

        one_result["score"] = person_to_joint_assoc[ridxPred, -2] * \
            person_to_joint_assoc[ridxPred, -1]
        one_result["keypoints"] = list(keypoints.reshape(51))
        #print(one_result)
        outputs.append(one_result)


def eval_coco(outputs, args, imgIds):
    from pycocotools.cocoeval import COCOeval
    """Evaluate images on Coco test set
    :param outputs: list of dictionaries, the models' processed outputs
    :param dataDir: string, path to the MSCOCO data directory
    :param imgIds: list, all the image ids in the validation set
    :returns : float, the mAP score
    """
    annType = 'keypoints'
    # initialize COCO ground truth api
    #dataType = 'val2014'
    #annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
    cocoGt = COCO(args.ann_dir)  # load annotations
    cocoDt = cocoGt.loadRes(args.result_json)  # load model outputs

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    #os.remove('results.json')
    # return Average Precision
    return cocoEval.stats[0]


if __name__ == "__main__":
    
    import cv2
    import os
    import argparse
    import torch
    import numpy as np
    from pycocotools.coco import COCO
    from collections import OrderedDict
    from .network import network_factory 
    
    #config parameters
    print("remember to check your forward return value of your new network")
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scale',default=[1],type=list)
    parser.add_argument('--heatmap_num',default=19,type=int)
    parser.add_argument('--paf_num',default=38,type=int)
    parser.add_argument('--weight_for_eval',default="./Pytorch_Pose_Estimation_Framework/ForSave/weight/op_new_ori/train_final.pth",
                        type=str)
    parser.add_argument('--eval_dir',default="./dataset/COCO/imgaes/",type=str)
    parser.add_argument('--ann_dir',default="./dataset/COCO/annotations/person_keypoints_val2017.json",type=str)
    parser.add_argument('--result_img_dir',default="./Pytorch_Pose_Estimation_Framework/ForSave/imgs/openpose3",
                        type=str)
    parser.add_argument('--result_json', default="./Pytorch_Pose_Estimation_Framework/ForSave/json/val_76_old.json",
                        type=str)
    parser.add_argument('--thre1', default=0.1, type=float)
    parser.add_argument('--thre2', default=0.05, type=float)
    parser.add_argument('--thre3', default=0.5, type=float)
    parser.add_argument('--pooling_factor', default=8, type=int)
    parser.add_argument("--base_size",default=368,type=int) 
    parser.add_argument('--filp', default=False, type=bool)                  
    parser.add_argument('--cal_score', default=True, type=bool)
    parser.add_argument('--preprocess', default=True, type=bool)
    parser.add_argument('--net_name',       default='CMU_new',                      type=str)
    network_factory.net_cli(parser,'CMU_new')
    args = parser.parse_args()

    outputs = []
    #load model
    model = network_factory.get_network(args)
    state_dict_ = torch.load(args.weight_for_eval)
    state_dict = state_dict_['model_state']
    try:
        model.load_state_dict(state_dict)
    except:
        new_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_dict[name] = v
        model.load_state_dict(new_dict)

    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()
    print("load model success")

    #load test data
    ann = COCO(args.ann_dir)
    catid =ann.getCatIds(catNms=['person'])
    eval_ids = ann.getImgIds(catIds=catid)
    #eval_ids = ann.getImgIds()
    
    #Start eval
    print("Processing Images in validation set"," total:",len(eval_ids))
    import matplotlib.pyplot as plt
    for ids in range(len(eval_ids)):
        if ids%10 == 0:
            print(ids)
        img = ann.loadImgs(eval_ids[ids])[0]
        file_name = img['file_name']
        file_path = os.path.join(args.eval_dir, file_name)
        oriImg = cv2.imread(file_path)

        pafs, heatmaps = Get_Multiple_outputs(oriImg,model,args.scale,args.filp,args.heatmap_num,args.paf_num,args)

        # # fordebug
        # plt.figure(num = 0, figsize = (15,4))
        # plt.subplot(121)
        # plt.imshow(pafs[:,:,0])
        # plt.subplot(122)
        # plt.imshow(heatmaps[:,:,10])
        # plt.savefig("test_heatmap_paf_add.png") #wrong 1 time two reasons
        # plt.show()
        # plt.figure(num = 1, figsize = (15,4))
        # plt.subplot(121)
        # pafs_all = np.zeros((pafs.shape[0],pafs.shape[1]))
        # for i in range(pafs.shape[2]):
        #     pafs_all = pafs_all + pafs[:,:,i] / pafs.shape[2]
        
        # heatmaps_all = np.zeros((heatmaps.shape[0],heatmaps.shape[1]))
        # for i in range(heatmaps.shape[2]):
        #     heatmaps_all = heatmaps_all + heatmaps[:,:,i] / heatmaps.shape[2]

        # plt.imshow(pafs_all[:,:])
        # plt.subplot(122)
        # plt.imshow(heatmaps_all[:,:])
        # plt.savefig("test_heatmap_paf_all.png") #wrong 1 time two reasons
        # plt.show()

        param = {'thre1': args.thre1, 'thre2': args.thre2, 'thre3': args.thre3}
        
        _, to_plot, candidate, subset = post.decode_pose(
            oriImg, param, heatmaps, pafs)
        vis_path = os.path.join(args.result_img_dir, file_name)
        cv2.imwrite(vis_path, to_plot)
        #print(subset)
        append_result(eval_ids[ids], subset, candidate, outputs)

    #print(outputs)
    with open(args.result_json, 'w') as f:
        json.dump(outputs, f)
    if args.cal_score:
        eval_coco(outputs=outputs, args=args, imgIds=eval_ids)
    print("finish")
    


    
    









    

