# Pytorch Pose Estimation Framework

A pytorch pose estimation framework using by myself for research.<br>
It contains a pytorch framework which is suitable for pose estimation.<br>
The main partion of it focus on OpenPose reproduction, with simliar mAP in their paper.<br>
You can also found code for my HSI paper in train_offset and train_mask.<br>


## Content

1. [Requirement](#Requirement)
2. [Test](#Test)
3. [Train](#Train)
4. [Result](#Result)
5. [Citation](#Citation)
6. [License](#License)



## Requirement

1. `Pytorch 1.2.0`
2. `Torchvision 0.4.0`
3. `TensorboardX`
This repro using the environment created by `Anaconda`, `cudatoolkits` and `cudnn` installed by conda automaticly.<br>



## Test

1. for test, download our pretrained model in [dropbox](), you also need parpare data by `bash get_data.sh`
2. open `evaluate.py`, all parameters related to evaluation are shown here.<br>
   `1160` means OpenPose offical small val dataset, which has been uesd in [OpenPose paper](https://arxiv.org/pdf/1611.08050.pdf).<br>
   `others` means COCO val 2017, which contains 2693 images or 5000 images, choosen by yourself.<br>
   `scale` should be `[0.5,1.0,1.5,2.0]` to get the maximum accuracy, as same as OpenPose
3. run `python -m Pytorch_Pose_Estimation_Framework.evaluate --val_type=1160 --network=CMU_old --scale=0.5,1.0,1.5,2.0` to get the mAP result
4. if you want to run yourselves images, just modify the `main` function in `evaluate.py`, actually, just some path need to be changed



## Train

1. all parameters related to train is in `train_op_baseline.py`
2. you need prepare the data by `bash get_data.sh`
3. you need generate mask file by running `generate_mask.py`
4. you need generate hdf5 file for training, about `200G` disk space is needed. run `generate_hdf5.py`
5. check the data path in `datasets/dataloader/cmu_h5_mainloader`
6. run `python -m Pytorch_Pose_Estimation_Framework.train_op_baseline --network=CMU_old `



## Result

We empirically trained the model for `55 epochs` and achieved comparable performance to the results reported in the original paper.<br> We also compared with the offical released caffe model which is by [Zhe Cao](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).<br>

|  Method   |   Validation   |   AP   |
|-----------|:--------------:|:------:|
|[Openpose paper](https://arxiv.org/pdf/1611.08050.pdf) |  COCO2014-Val-1k   |    58.4   | 
|[Openpose model]()|    COCO2014-Val-1k   |    56.3   |
|    This repo    |    COCO2014-Val-1k    |    58.4   |


## Acknowledgment

This repo is based upon[@kevinlin311tw](https://github.com/kevinlin311tw/keras-openpose-reproduce/blob/master/README.md) and [@tensorboy](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation).<br>
Thanks `kevinlin311tw` who is really nice to communicate


## Citation

Please cite the paper in your publications if it helps your research:

    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
      }
      
    @inproceedings{liu2020resolution,
        title={Resolution Irrelevant Encoding and Difficulty Balanced Loss Based Network Independent Supervision for Multi-Person Pose Estimation},
        author={Liu, Haiyang and Luo, Dingli and Du, Songlin and Ikenaga, Takeshi},
        booktitle={2020 13th International Conference on Human System Interaction (HSI)},
        pages={112--117},
        year={2020},
        organization={IEEE}
      }
      






