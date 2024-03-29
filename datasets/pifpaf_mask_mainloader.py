# ------------------------------------------------------------------------------
# The PifPaf_mask 53K dataloader of total framework
# base on https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation
# Modify by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

import copy
import logging
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist
from pycocotools.coco import maskUtils

from .encoder import heatmap2,paf2,utils,transforms

def loader_cli(parser):
    ''' 
    some parameters of dataloader
    1. data path
    2. training img size
    3. training img number
    4. some augment setting
    '''
    print('choose PifPaf_mask 53K data, loading parameters...')
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train_ann_dir',       default='./dataset/COCO/annotations/person_keypoints_train2017.json')
    group.add_argument('--train_image_dir',     default='./dataset/COCO/images/train2017')
    group.add_argument('--val_ann_dir',         default='./dataset/COCO/annotations/person_keypoints_val2017.json')
    group.add_argument('--val_image_dir',       default='./dataset/COCO/images/val2017')
    group.add_argument('--n_images',            default=None, type=int,
                       help='number of images to sample')
    group.add_argument('--loader_workers',      default=16, type=int,
                       help='number of workers for data loading')
    group.add_argument('--square_edge',         default=480, type=int,
                        help='square edge of input images')
    group.add_argument('--no_augmentation',     default=False, action='store_false',
                        help='do not apply data augmentation')

class Meta(object):
    """
    Metadata representing a single data point for training.
    """
    __slots__ = (
        'image_id',
        'height',
        'width',
        'keypoints',
        'masks_segments',
        'mask'
        )

    def __init__(self, image_id,height,width):

        self.image_id = image_id
        self.height = height
        self.width = width
        self.keypoints = None
        self.masks_segments = None 
        self.mask = None
       

class COCOKeypoints(Dataset):
    ''' 
    finish generate mask and gt for data 
    '''

    def __init__(self,ann_path, img_path, augment=None,
                 other_aug=None, n_images=None, all_images=False, all_persons=False,
                 input_y=480, input_x=480, stride=4):

        self.root = img_path
        self.coco = COCO(ann_path)       
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        #self.ids
        if all_images:
            self.ids = self.coco.getImgIds()
        elif all_persons:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
        else:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
            self.filter_for_keypoint_annotations()
        if n_images:
            self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        self.preprocess = augment 
        self.image_transform = other_aug or transforms.image_transform
        self.HEATMAP_COUNT = len(get_keypoints())
        self.LIMB_IDS = kp_connections(get_keypoints())
        self.input_y = input_y
        self.input_x = input_x        
        self.stride = stride
        #self.all_meta = None

        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        '''
        filter for the:
        1. no keypoints annotations
        2. keypoints annotatios visiable are all zero
        '''
        print('filter for keypoint annotations ...')
        all_meta = []
        def has_keypoint_annotation(image_id,all_meta):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            img_meta = self.coco.imgs[image_id]
            img_h = img_meta['height']
            img_w = img_meta['width']
            '''
            have person already
            for each person check,
            '''
            persons = []
            prev_center = []
            masks = []
            keypoints = []

            persons_ids = persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')
            
            for id in list(persons_ids):
                person_meta = anns[id]

                if person_meta["iscrowd"]:
                    masks.append(self.coco.annToRLE(person_meta))
                    continue

                # skip this person if parts number is too low or if
                # segmentation area is too small

                if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
                    masks.append(self.coco.annToRLE(person_meta))
                    continue

                person_center = [person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
                                    person_meta["bbox"][1] + person_meta["bbox"][3] / 2]

                # skip this person if the distance to existing person is too small

                too_close = False
                for pc in prev_center:
                    a = np.expand_dims(pc[:2], axis=0)
                    b = np.expand_dims(person_center, axis=0)
                    dist = cdist(a, b)[0]
                    if dist < pc[2]*0.3:
                        too_close = True
                        break

                if too_close:
                    # add mask of this person. we don't want to show the network
                    # unlabeled people
                    masks.append(self.coco.annToRLE(person_meta))
                    continue

                pers = Meta(
                        image_id = image_id,
                        height = img_h,
                        width = img_w)

                keypoints.append(person_meta["keypoints"])
                persons.append(pers)
                prev_center.append(np.append(person_center, max(person_meta["bbox"][2],
                                                                person_meta["bbox"][3])))
            if len(persons) > 0:
                    main_person = persons[0]
                    main_person.masks_segments = masks
                    main_person.keypoints = keypoints
                    #main_person.all_joints = JointsLoader.from_coco_keypoints(keypoints, w, h)
                    all_meta.append(main_person)
                    return True

            # for ann in anns:
            #     if 'keypoints' not in ann:
            #         continue
            #     if any(v > 0.0 for v in ann['keypoints'][2::3]):
            #         return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id,all_meta)]
        self.all_meta = all_meta
        
        print('... done.')

    def add_neck(self, keypoint):
        '''
        MS COCO annotation order:
        0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
        5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
        9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
        14: r knee		15: l ankle		16: r ankle
        The order in this work:
        (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
        5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
        9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
        13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
        17-'left_ear' )
        '''
        our_order = [0, 17, 6, 8, 10, 5, 7, 9,
                     12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # Index 6 is right shoulder and Index 5 is left shoulder
        right_shoulder = keypoint[6, :]
        left_shoulder = keypoint[5, :]
        neck = (right_shoulder + left_shoulder) / 2
        if right_shoulder[2] == 2 and left_shoulder[2] == 2:
            neck[2] = 2
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]

        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        keypoint = np.vstack((keypoint, neck))
        keypoint = keypoint[our_order, :]

        return keypoint
                
    def get_ground_truth(self, anns):
    
        grid_y = int(self.input_y / self.stride)
        grid_x = int(self.input_x / self.stride)
        channels_heat = (self.HEATMAP_COUNT + 1)
        channels_paf = 2 * len(self.LIMB_IDS)
        heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
        pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))

        prev_center = []
        keypoints_ann = []
        persons_ids = persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')
        
        for id in list(persons_ids):
            person_meta = anns[id]

            if person_meta["iscrowd"]:
                continue

            if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
                continue

            person_center = [person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
                                person_meta["bbox"][1] + person_meta["bbox"][3] / 2]

            too_close = False
            for pc in prev_center:
                a = np.expand_dims(pc[:2], axis=0)
                b = np.expand_dims(person_center, axis=0)
                dist = cdist(a, b)[0]
                if dist < pc[2]*0.3:
                    too_close = True
                    break

            if too_close:
                continue

            keypoints_ann.append(person_meta["keypoints"])
            prev_center.append(np.append(person_center, max(person_meta["bbox"][2],
                                                            person_meta["bbox"][3])))

        keypoints = []
        for per in range(len(keypoints_ann)):
            single_keypoints = np.array(keypoints_ann[per]).reshape(17,3)
            single_keypoints = self.add_neck(single_keypoints)
            keypoints.append(single_keypoints)
        keypoints = np.array(keypoints)
        keypoints = self.remove_illegal_joint(keypoints)

        # for ann in anns:
        #     single_keypoints = np.array(ann['keypoints']).reshape(17,3)
        #     single_keypoints = self.add_neck(single_keypoints)
        #     keypoints.append(single_keypoints)
        # keypoints = np.array(keypoints)
        # keypoints = self.remove_illegal_joint(keypoints)

        # confidance maps for body parts
        for i in range(self.HEATMAP_COUNT):
            joints = [jo[i] for jo in keypoints]
            for joint in joints:
                if joint[2] > 0.5:
                    center = joint[:2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = heatmap2.putGaussianMaps(
                        center, gaussian_map,
                        7.0, grid_y, grid_x, self.stride)
        # pafs
        for i, (k1, k2) in enumerate(self.LIMB_IDS):
            # limb
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            for joint in keypoints:
                if joint[k1, 2] > 0.5 and joint[k2, 2] > 0.5:
                    centerA = joint[k1, :2]
                    centerB = joint[k2, :2]
                    vec_map = pafs[:, :, 2 * i:2 * (i + 1)]

                    pafs[:, :, 2 * i:2 * (i + 1)], count = paf2.putVecMaps(
                        centerA=centerA,
                        centerB=centerB,
                        accumulate_vec_map=vec_map,
                        count=count, grid_y=grid_y, grid_x=grid_x, stride=self.stride
                    )

        # background
        heatmaps[:, :, -1] = np.maximum(
            1 - np.max(heatmaps[:, :, :self.HEATMAP_COUNT], axis=2),
            0.
        )
        return heatmaps, pafs
    
    def generate_apply_mask(self,meta,image):      
        if meta.masks_segments:
            mask_miss = np.ones((meta.height, meta.width), dtype=np.uint8)
            for seg in meta.masks_segments:
                bin_mask = maskUtils.decode(seg)
                bin_mask = np.logical_not(bin_mask)
                mask_miss = np.bitwise_and(mask_miss, bin_mask)
            meta.mask = mask_miss
        
        if meta.mask is not None:
            image = np.array(image,dtype=np.uint8)
            image[:, :, 0] = image[:, :, 0] * meta.mask
            image[:, :, 1] = image[:, :, 1] * meta.mask
            image[:, :, 2] = image[:, :, 2] * meta.mask
            #image = Image.fromarray(meta.mask.astype(np.uint8))
            image = Image.fromarray(image.astype(np.uint8))
        return image
            

    def remove_illegal_joint(self, keypoints):
        
        if len(keypoints) != 0:
            MAGIC_CONSTANT = (-1, -1, 0)
            mask = np.logical_or.reduce((keypoints[:, :, 0] >= self.input_x,
                                        keypoints[:, :, 0] < 0,
                                        keypoints[:, :, 1] >= self.input_y,
                                        keypoints[:, :, 1] < 0))
            keypoints[mask] = MAGIC_CONSTANT

        return keypoints

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        """
        per_meta = self.all_meta[index]
        
        ann_ids = self.coco.getAnnIds(imgIds=per_meta.image_id)
        anns = self.coco.loadAnns(ann_ids)
        #why?
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(per_meta.image_id)[0]
        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta_init = {
            'dataset_index': index,
            'image_id': per_meta.image_id,
            'file_name': image_info['file_name'],
        }
        # finish generate mask
        image = self.generate_apply_mask(per_meta,image)
        image, anns, meta = self.preprocess(image, anns, None)
        #mask, anns, meta = self.preprocess(mask, anns, None)
             
        if isinstance(image, list):
            return self.multi_image_processing(image, anns, meta,meta_init)

        return self.single_image_processing(image, anns, meta,meta_init)

    def multi_image_processing(self, image_list, anns_list, meta_list, meta_init):
        return list(zip(*[
            self.single_image_processing(image, anns, meta, meta_init)
            for image, anns, meta in zip(image_list, anns_list, meta_list)
        ]))

    def single_image_processing(self, image, anns, meta, meta_init):
        meta.update(meta_init)

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        self.log.debug(meta)

        heatmaps, pafs = self.get_ground_truth(anns)
        
        heatmaps = torch.from_numpy(
            heatmaps.transpose((2, 0, 1)).astype(np.float32))
            
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))       
        return image, heatmaps, pafs

    def __len__(self):
        return len(self.ids)

def train_factory(type_,args):
    ''' return train or val or pertrain data '''
   
    if type_ == "train":
        preprocess = transforms.Compose([
            transforms.Normalize(),
            transforms.RandomApply(transforms.HFlip(), 0.5),
            transforms.RescaleRelative(),
            transforms.Crop(args.square_edge),
            transforms.CenterPad(args.square_edge)])
        data_ = COCOKeypoints(args.train_ann_dir, args.train_image_dir,
                              augment=preprocess, input_x=args.square_edge, input_y=args.square_edge,
                              other_aug=transforms.image_transform_train, n_images=args.n_images)
                  
        train_loader = DataLoader(data_, shuffle=True, batch_size=args.batch_size,
                                 num_workers=args.loader_workers, #collate_fn=collate_images_anns_meta,
                                 pin_memory=True, drop_last=True)
        return train_loader
    else:
        preprocess = transforms.Compose([
            transforms.Normalize(),
            # transforms.RandomApply(transforms.HFlip(), 0.5),
            # transforms.RescaleRelative(),
            transforms.Crop(args.square_edge),
            transforms.CenterPad(args.square_edge)])

        data_ = COCOKeypoints(args.val_ann_dir, args.val_image_dir, augment=preprocess, 
                              input_x=args.square_edge, input_y=args.square_edge)

        val_loader = DataLoader(data_, shuffle=False, batch_size=args.batch_size,
                                 num_workers=args.loader_workers, 
                                 pin_memory=True, drop_last=False)
        return val_loader

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('neck'), keypoints.index('right_hip')],  
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('neck'), keypoints.index('left_hip')],                
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],          
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],     
        [keypoints.index('right_shoulder'), keypoints.index('right_eye')],        
        [keypoints.index('neck'), keypoints.index('left_shoulder')], 
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_eye')],               
        [keypoints.index('neck'), keypoints.index('nose')],                      
        [keypoints.index('nose'), keypoints.index('right_eye')],
        [keypoints.index('nose'), keypoints.index('left_eye')],        
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')]
    ]
    return kp_lines
    
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',   
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_eye',                                                                    
        'left_eye',
        'right_ear',
        'left_ear']

    return keypoints    

