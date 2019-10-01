# ------------------------------------------------------------------------------
# The newest OpenPose Pytorch Implementation 
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import init


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def cli(parser):
    ''' network config
        1. paf and heatmap nums
        2. weight path
    '''

    group = parser.add_argument_group('network')
    group.add_argument('--heatmap_num', default=19, type=int)
    group.add_argument('--paf_num', default=38, type=int)
    group.add_argument('--paf_stage', default=4, type=int)
    group.add_argument('--weight_vgg19', default='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
    
class CMUnetwork(nn.Module):
    ''' the newest cmu network'''

    def __init__ (self,args):
        # already finish the init_weight in each block
        super(CMUnetwork, self).__init__()
        self.state_0 = VGG_block()
        self.state_1 = state_1_block(128, args.paf_num)
        self.state_2 = state_n_block(128+args.paf_num,args.paf_num)
        self.state_3 = state_n_block(128+args.paf_num,args.paf_num)
        self.state_4 = state_n_block(128+args.paf_num,args.paf_num)
        self.state_5 = state_1_block(128+args.paf_num,args.heatmap_num)
        self.state_6 = state_n_block(128+args.heatmap_num,args.heatmap_num)

    def forward(self,input_0):

        saved_for_loss = []

        output_0 = self.state_0(input_0)
        output_1 = self.state_1(output_0)
        input_2  = torch.cat([output_1,output_0],1)
        saved_for_loss.append(output_1)

        output_2 = self.state_2(input_2)
        input_3  = torch.cat([output_2,output_0],1)
        saved_for_loss.append(output_2)

        output_3 = self.state_3(input_3)
        input_4  = torch.cat([output_3,output_0],1)
        saved_for_loss.append(output_3)

        output_4 = self.state_4(input_4)
        input_5  = torch.cat([output_4,output_0],1)
        saved_for_loss.append(output_4)

        output_5 = self.state_5(input_5)
        input_6  = torch.cat([output_5,output_0],1)
        saved_for_loss.append(output_5)

        output_6 = self.state_6(input_6)
        saved_for_loss.append(output_6)

        return (output_4,output_6), saved_for_loss


class dense_block(nn.Module):
    '''1. basic dense block of the new openpose
       2. add conv1,2,3 output together to output
       3. default kernal_size = 3,bias = true
    '''

    def __init__(self, in_dim, out_dim):
        super(dense_block, self).__init__()
        # default inplace = False for ReLU
        
        self.conv1 = nn.Sequential( #nn.Conv2d(in_dim, 128, 1, 1, 0),
                                    nn.Conv2d(in_dim, 128, 3, 1, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace = True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace = True))
        # self.conv3 = nn.Sequential(nn.Conv2d(128, (out_dim-256), 3, 1, 1),
        #                             nn.ReLU(inplace = True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace = True))
                                   
        #self.conv4 = nn.Sequential(nn.Conv2d(384, out_dim, 1, 1, 0),nn.ReLU(inplace = True))
        self.initialize_weight()
        

    def forward(self,input_1):
        # debug = True
        output_1 = self.conv1(input_1)
        output_2 = self.conv2(output_1)
        output_3 = self.conv3(output_2)
        output = torch.cat([output_1,output_2,output_3],1)
        #output_4 = self.conv4(output)
        # if debug:
        #     output = output_3
        # output = torch.cat([output_1,output_2,output_3],1)
        return output
    
    def initialize_weight(self):
        for m in self.modules():
            #print('need check init')
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

class dense_block_0(nn.Module):
    '''1. basic dense block of the new openpose
       2. add conv1,2,3 output together to output
       3. default kernal_size = 3,bias = true
    '''

    def __init__(self, in_dim, out_dim):
        super(dense_block_0, self).__init__()
        # default inplace = False for ReLU
        
        self.conv1 = nn.Sequential( #nn.Conv2d(in_dim, 128, 1, 1, 0),
                                    nn.Conv2d(in_dim, 96, 3, 1, 1),
                                    nn.BatchNorm2d(96),
                                    nn.ReLU(inplace = True))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1),
                                   nn.BatchNorm2d(96), 
                                    nn.ReLU(inplace = True))
        # self.conv3 = nn.Sequential(nn.Conv2d(128, (out_dim-256), 3, 1, 1),
        #                             nn.ReLU(inplace = True))
        self.conv3 = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1), nn.BatchNorm2d(96),nn.ReLU(inplace = True))
        #self.conv4 = nn.Sequential(nn.Conv2d(384, out_dim, 1, 1, 0),nn.ReLU(inplace = True))
        self.initialize_weight()
        

    def forward(self,input_1):
        # debug = True
        output_1 = self.conv1(input_1)
        output_2 = self.conv2(output_1)
        output_3 = self.conv3(output_2)
        output = torch.cat([output_1,output_2,output_3],1)
        #output_4 = self.conv4(output)
        # if debug:
        #     output = output_3
        # output = torch.cat([output_1,output_2,output_3],1)
        return output
    
    def initialize_weight(self):
        for m in self.modules():
            #print('need check init')
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    
    
        
class state_n_block(nn.Module):
    ''' 1. for state 1 in_dim = 128
        2. for other state depend on the paf and heatmap channels
    '''

    def __init__(self, in_dim, out_dim):
        # 384 = 128 *3
        super(state_n_block, self).__init__()
        self.block1 = dense_block(in_dim,384)
        self.block2 = dense_block(384,128)
        self.block3 = dense_block(384,128)
        self.block4 = dense_block(384,128)
        self.block5 = dense_block(384,128)
        self.conv1  = nn.Sequential(nn.Conv2d(384, 512, 1, 1, 0),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace = True))
        self.conv2  = nn.Conv2d(512,out_dim,1,1,0)
        self.initialize_weight()

    def forward(self,input_1):
        ''' inplace the midresult '''
        output_1 = self.block1(input_1)
        output_1 = self.block2(output_1)
        output_1 = self.block3(output_1)                     
        output_1 = self.block4(output_1)
        output_1 = self.block5(output_1)
        output_1 = self.conv1(output_1)
        output_1 = self.conv2(output_1)
        return output_1
    
    def initialize_weight(self):
        '''init 1*1 conv block
        '''
        init.normal_(self.conv1[0].weight, std =0.01)
        init.constant_(self.conv1[0].bias, 0.0)
        init.normal_(self.conv2.weight, std =0.01)
        init.constant_(self.conv2.bias, 0.0)

class state_1_block(nn.Module):
    ''' 1. for state 1 in_dim = 128
        2. for other state depend on the paf and heatmap channels
    '''

    def __init__(self, in_dim, out_dim):
        # 384 = 128 *3
        super(state_1_block, self).__init__()
        self.block1 = dense_block_0(in_dim,288)
        self.block2 = dense_block_0(288,96)
        self.block3 = dense_block_0(288,96)
        self.block4 = dense_block_0(288,96)
        self.block5 = dense_block_0(288,96)
        self.conv1  = nn.Sequential(nn.Conv2d(288, 256, 1, 1, 0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace = True))
        self.conv2  = nn.Conv2d(256,out_dim,1,1,0)
        self.initialize_weight()

    def forward(self,input_1):
        ''' inplace the midresult '''
        output_1 = self.block1(input_1)
        output_1 = self.block2(output_1)
        output_1 = self.block3(output_1)                     
        output_1 = self.block4(output_1)
        output_1 = self.block5(output_1)
        output_1 = self.conv1(output_1)
        output_1 = self.conv2(output_1)
        return output_1

    def initialize_weight(self):
        '''init 1*1 conv block
        '''
        init.normal_(self.conv1[0].weight, std =0.01)
        init.constant_(self.conv1[0].bias, 0.0)
        init.normal_(self.conv2.weight, std =0.01)
        init.constant_(self.conv2.bias, 0.0)


class VGG_block(nn.Module):
    ''' 1. default have the bias
        2. using ReLU and 3 * max pooling
        3. 10 layers of VGG original
        4. 2 extra layers by CMU
        5. default in_dim = 3,out_dim = 128
        6. all kernal_size = 3, stride = 1
    '''

    def __init__(self, in_dim = 3, out_dim = 128):
        super(VGG_block, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1_1 = nn.ReLU(inplace = True)            
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.ReLU(inplace = True)
        self.pool_1 = nn.MaxPool2d(2, 2, 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2_1 = nn.ReLU(inplace = True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.ReLU(inplace = True)
        self.pool_2 = nn.MaxPool2d(2, 2, 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu3_1 = nn.ReLU(inplace = True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(inplace = True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.ReLU(inplace = True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_4 = nn.ReLU(inplace = True)
        self.pool_3 = nn.MaxPool2d(2, 2, 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.relu4_1 = nn.ReLU(inplace = True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU(inplace = True)
        self.conv4_3_cmu = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu4_3 = nn.ReLU(inplace = True)
        self.conv4_4_cmu = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu4_4 = nn.ReLU(inplace = True)

        self.initilization()
                                
    def forward(self,input_1):
        '''inplace middle result '''
        #print("before_vgg",input_1.size())
        output_1 = self.conv1_1(input_1)
        output_1 = self.relu1_1(output_1)
        output_1 = self.conv1_2(output_1)
        output_1 = self.relu1_2(output_1)
        output_1 = self.pool_1(output_1)                     
        output_1 = self.conv2_1(output_1)
        output_1 = self.relu2_1(output_1)
        output_1 = self.conv2_2(output_1)
        output_1 = self.relu2_2(output_1)
        output_1 = self.pool_2(output_1)
        output_1 = self.conv3_1(output_1)
        output_1 = self.relu3_1(output_1)
        output_1 = self.conv3_2(output_1)
        output_1 = self.relu3_2(output_1)
        output_1 = self.conv3_3(output_1)
        output_1 = self.relu3_3(output_1)
        output_1 = self.conv3_4(output_1)
        output_1 = self.relu3_4(output_1)
        output_1 = self.pool_3(output_1)
        output_1 = self.conv4_1(output_1)
        output_1 = self.relu4_1(output_1)
        output_1 = self.conv4_2(output_1)
        output_1 = self.relu4_2(output_1)
        output_1 = self.conv4_3_cmu(output_1)
        output_1 = self.bn1(output_1)
        output_1 = self.relu4_3(output_1)
        output_1 = self.conv4_4_cmu(output_1)
        output_1 = self.bn2(output_1)
        output_1 = self.relu4_4(output_1)
        #print("after_vgg",input_1.size())
        return output_1
    
    def initilization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  
                    init.constant_(m.bias, 0.0)

 

    






    
