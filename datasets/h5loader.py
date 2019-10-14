
import h5py
import random
import json
import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset

from .py_rmpe_server.py_rmpe_config import RmpeGlobalConfig, RmpeCocoConfig
from .py_rmpe_server.py_rmpe_transformer import Transformer, AugmentSelection
from .py_rmpe_server.py_rmpe_heatmapper import Heatmapper

class h5loader(Dataset):

    def __init__(self, h5file, augment = True):

        self.h5file = h5file
        self.h5 = h5py.File(self.h5file, "r")
        self.datum = self.h5['datum']
        self.heatmapper = Heatmapper()
        self.augment = augment
        self.split_point = 38
        self.vec_num = 38
        self.heat_num = 19
        self.keys = list(self.datum.keys())
    
    def __getitem__(self, index):
        #print('')
        key = self.keys[index]
        import matplotlib.pyplot as plt
        import cv2

        image, mask, meta = self.read_data(key)
        image, mask, _, labels = self.transform_data(image, mask, meta)
        #imagefix = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # fig = plt.figure()
        # a = fig.add_subplot(2,2,1)
        # a.set_title('ori_image')
        # plt.imshow(imagefix)
        
        image = image.astype(np.float32)
        image = image / 256. - 0.5
        image = np.transpose(image,(2, 0, 1))
        image = torch.from_numpy(image)
       
        #image = image.astype(np.float32)
        # image = image / 256. - 0.5
        # image = image.transpose((2, 0, 1)).astype(np.float32)


        vec_weights = np.repeat(mask[:,:,np.newaxis], self.vec_num, axis=2)
        heat_weights = np.repeat(mask[:,:,np.newaxis], self.heat_num, axis=2)
        vec_label = labels[:self.split_point, :, :]
        heat_label = labels[self.split_point:, :, :]
        vec_label = torch.from_numpy(vec_label)#.astype(np.float32)
        heat_label = torch.from_numpy(heat_label)#.astype(np.float32)
        heat_weights = torch.from_numpy(heat_weights)#.astype(np.float32)
        vec_weights = torch.from_numpy(vec_weights)#.astype(np.float32)
        vec_weights = np.transpose(vec_weights,(2, 0, 1))
        heat_weights = np.transpose(heat_weights,(2, 0, 1))
        vec_label = vec_label.type(torch.float32)
        heat_label = heat_label.type(torch.float32)
        heat_weights = heat_weights.type(torch.float32)
        vec_weights = vec_weights.type(torch.float32)
        #print(vec_label.size())
        
        # image = image.numpy()
        # image = np.transpose(image,(1,2,0))
        # image = image + 0.5
        # image = image * 256. 
        # image = image.astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # fig = plt.figure()
        # a = fig.add_subplot(2,2,2)
        # a.set_title('ori_image2')
        # plt.imshow(image)
        # plt.show()

        return image, heat_label, heat_weights, vec_label, vec_weights

    def __len__(self):
        return len(list(self.datum.keys()))

    def read_data(self, key):

        entry = self.datum[key]

        assert 'meta' in entry.attrs, "No 'meta' attribute in .h5 file. Did you generate .h5 with new code?"

        meta = json.loads(entry.attrs['meta'])
        meta['joints'] = RmpeCocoConfig.convert(np.array(meta['joints']))
        data = entry.value

        if data.shape[0] <= 6:
            # TODO: this is extra work, should write in store in correct format (not transposed)
            # can't do now because I want storage compatibility yet
            # we need image in classical not transposed format in this program for warp affine
            data = data.transpose([1,2,0])

        img = data[:,:,0:3]
        mask_miss = data[:,:,4]
        mask = data[:,:,5]

        return img, mask_miss, meta

    def transform_data(self, img, mask,  meta):

        aug = AugmentSelection.random() if self.augment else AugmentSelection.unrandom()
        img, mask, meta = Transformer.transform(img, mask, meta, aug=aug)
        labels = self.heatmapper.create_heatmaps(meta['joints'], mask)

        return img, mask, meta, labels

    def __del__(self):
        self.h5.close()


def train_factory(type_,args):
    ''' return train or val or pertrain data '''
    #print('useme'
    print('start loading')
    if type_ == "train":
        data_ = h5loader('/home/liuhaiyang/keras-openpose-reproduce/dataset/train_dataset_2014.h5')
        print('init data success')         
        train_loader = DataLoader(data_, shuffle=True, batch_size=10,
                                 num_workers=1, #collate_fn=collate_images_anns_meta,
                                 pin_memory=True, drop_last=True)
        print('train dataset len:{}'.format(len(train_loader.dataset)))
        return train_loader
    else:
        data_ = h5loader('/home/liuhaiyang/keras-openpose-reproduce/dataset/val_dataset_2014.h5')

        val_loader = DataLoader(data_, shuffle=False, batch_size=10,
                                 num_workers=1, 
                                 pin_memory=True, drop_last=False)
        print('val dataset len:{}'.format(len(val_loader.dataset)))
        return val_loader