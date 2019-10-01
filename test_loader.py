
from .datasets import mainloader
import argparse
import matplotlib.pyplot as plt



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

def main():
    args = cli()

    train_loader = mainloader.train_factory('train',args)
    val_loader = mainloader.train_factory('val',args)

    for batch_id, (img,heatmap_target,paf_target) in enumerate(train_loader):
        for img_id in range(args.batch_size):
            np_single_img = img[img_id,:,:,:].numpy()
            np_heatmap = heatmap_target[img_id,0:1,:,:].numpy()
            np_paf = paf_target[img_id,0:1,:,:].numpy()

            plt.figure(num = 0, figsize = (15,4))
            plt.subplot(221)
            plt.imshow(np_single_img)
            plt.subplot(222)
            plt.imshow(np_heatmap)
            plt.subplot(223)
            plt.imshow(np_paf)
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
            plt.show()


if __name__ == "__main__":
    main()
