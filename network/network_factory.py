# ------------------------------------------------------------------------------
# The network factory of total framework 
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

from .openpose import CMU_BN_net,CMU_old,CMUnet
from .self import resnet_op,baseline_old,baseline_op

def net_cli(parser,net_name):
    
    net_LUT_cli = {
        'CMU_bn_new':       CMU_BN_net.network_cli,
        'CMU_old':          CMU_old.network_cli,
        'CMU_new':          CMUnet.network_cli,
        'fpn':              resnet_op.network_cli,
        'sample_baseline':  baseline_old.network_cli,
        'sample_baseline1': baseline_old.network_cli,
    }
    try:
        net_setting = net_LUT_cli.get(net_name)
        net_setting(parser)
    except:
        print("network cli error, pls open network factory choose right network")

def get_network(args):

    net_name = args.net_name
    net_LUT_factory = {
        'CMU_bn_new':       CMU_BN_net.CMUnetwork(args),
        'CMU_old':          CMU_old.CMUnetwork(args),
        'CMU_new':          CMUnet.CMUnetwork(args),
        'fpn':              resnet_op.FPN50,
        'sample_baseline':  baseline_old.baselinenet(),
        'sample_baseline1': baseline_old.baselinenet(),
    }
    try:
        if net_name == 'fpn':
            net_function = net_LUT_factory.get(net_name)
            net_instance = net_function()
        else:
            net_instance = net_LUT_factory.get(net_name)
        return net_instance
    except:
        print("network tpye error, pls open network factory choose right network")