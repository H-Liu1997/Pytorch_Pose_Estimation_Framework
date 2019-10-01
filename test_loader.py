# ------------------------------------------------------------------------------
# The test portion of dataloader 
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

from .datasets import mainloader
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2


def cli():
    """
    set all parameters 
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mainloader.train_cli(parser)
    parser.add_argument('--batch_size', default= 5, type=int,
                        help='batch size per gpu')
 
    args = parser.parse_args()
    return args

def _get_bgimg(inp, target_size=None):
    if target_size:
        inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
    return inp

def inverse_vgg_preprocess(image):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    image = image.transpose((1,2,0))
    
    for i in range(3):
        image[:, :, i] = image[:, :, i] * stds[i]
        image[:, :, i] = image[:, :, i] + means[i]
    image = image.copy()[:,:,::-1]
    image = image*255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def main():
    args = cli()

    train_loader = mainloader.train_factory('train',args)
    val_loader = mainloader.train_factory('val',args)

    for batch_id, (img,heatmap_target,paf_target) in enumerate(train_loader):
        for img_id in range(args.batch_size):
            
            np_single_img = img[img_id,:,:,:].numpy()
            np_single_img = inverse_vgg_preprocess(np_single_img)
            np_heatmap = heatmap_target[img_id,0:18,:,:].numpy().transpose((1, 2, 0))
            np_paf = paf_target[img_id,0:37,:,:].numpy()

            fig = plt.figure()
            a = fig.add_subplot(2,2,1)
            a.set_title('ori_image')
            plt.imshow(_get_bgimg(np_single_img))

            a = fig.add_subplot(2,2,2)
            a.set_title('heatmap')
            plt.imshow(_get_bgimg(np_single_img, target_size=(46, 46)),alpha=0.7)
            tmp = np.amax(np_heatmap, axis=2)
            plt.imshow(tmp, cmap=plt.cm.Reds, alpha=0.3)
            plt.colorbar()

            tmp2_odd = np.amax(np.absolute(np_paf[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(np_paf[1::2, :, :]), axis=0)

            a = fig.add_subplot(2, 2, 3)
            a.set_title('paf-x')
            plt.imshow(_get_bgimg(np_single_img, target_size=(46,46)), alpha=0.7)
            plt.imshow(tmp2_odd, cmap=plt.cm.Reds, alpha=0.3)
            plt.colorbar()

            a = fig.add_subplot(2, 2, 4)
            a.set_title('paf-y')
            plt.imshow(_get_bgimg(np_single_img, target_size=(46,46)), alpha=0.7)
            plt.imshow(tmp2_even, cmap=plt.cm.Reds, alpha=0.3)
            plt.colorbar()

            plt.show()

if __name__ == "__main__":
    main()
