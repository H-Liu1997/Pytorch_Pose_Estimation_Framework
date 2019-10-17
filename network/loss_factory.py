# ------------------------------------------------------------------------------
# The loss factory of total framework 
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

from .openpose import CMUnet_loss
from .self import resnet_loss


def loss_cli(parser,loss_name):
    
    loss_LUT_cli = {
        'CMU_2branch':      CMUnet_loss.loss_cli,
        'CMU_2b_mask':      CMUnet_loss.loss_cli,
        'CMU_1branch':      CMUnet_loss.loss_cli,
        'fpn':              resnet_loss.loss_cli,
        'sample_baseline':  resnet_loss.loss_cli,
    }
    try:
        loss_setting = loss_LUT_cli.get(loss_name,other)
        loss_setting(parser,loss_name)
    except:
        print("loss tpye error, pls open loss factory choose right loss")

def get_loss_function(args):

    loss_name = args.loss_name
    loss_LUT = {
        'CMU_2branch':      CMUnet_loss.get_old_loss,
        'CMU_2b_mask':      CMUnet_loss.get_mask_loss,
        'CMU_1branch':      CMUnet_loss.get_loss,
        'fpn':              resnet_loss.get_loss,
        'sample_baseline':  resnet_loss.get_loss,
    }
    try:
        loss_function = loss_LUT.get(loss_name,other)
        return loss_function
    except:
        print("loss tpye error, pls open loss factory choose right loss")