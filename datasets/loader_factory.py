from . import cmu_mainloader

def loader_factory(args):
    '''
    CMU: 1. 120K data for training, generated by train2014+val2014.
         2. 4K for val, generated by first 2.6k images in val2014.
         3. haven't masked the train img, mask when calculte loss.
         4. mask condition, iscrowd, size<32*32,joints<5 and too close.
    OpenPifPaf:
         1. 64k,56k,117k data for training depend on your choose in train2017
         2. 2.6K or 5K for val depend on choose in val2014
         3. no mask
    PifPaf_mask:
         1. base on OpenPifPaf and add mask for bad person annotations
         2. currently only mask before training, not good.
         3. after fliter image from 56k to 53k, mAP decrease 1.% 
    CMU_single:
         1. 55k for trianing, generated by train2017
         2. one image only using one time.       
    '''
    loader_name = args.loader
    if loader_name == 'CMU':  
        print('choose CMU 120k/4k data, load parameters...')
        cmu_mainloader.loader_cli(args)
        print('train img size: {}'.format(args.img_size))
        print('processing method: {}'.format(args.process))
        train_factory_cmu = cmu_mainloader.train_factory
        return train_factory_cmu
    elif loader_name == 'OpenPifPaf':
        pass
    elif loader_name == 'CMU_single':
        pass
    else: print('loader name error, please choose CMU, OpenPifPaf or CMU_single')    
