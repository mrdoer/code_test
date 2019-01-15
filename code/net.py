# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-05 11:16:24
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-19 22:11:49
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

class DenseSiameseRPN(nn.Module):
    def __init__(self):
        super(DenseSiameseRPN,self).__init__()
        self.features = RDN()

        self.k = 5
        self.s = 4
        self.conv1 = nn.Conv2d(256, 2*self.k*256, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4*self.k*256, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(256, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(256, 4* self.k, kernel_size = 4, bias = False)

    def forward(self, template, detection):
            """
            把template的类别,坐标的特征作为检测cconv和rconv的检测器
            把ckernel, rkernel转换到cconv, rconv
            """
            template = self.features(template)
            detection = self.features(detection)

            ckernal = self.conv1(template)
            ckernal = ckernal.view(2* self.k, 256, 7, 7)
            cinput  = self.conv3(detection)


            rkernal = self.conv2(template)
            rkernal = rkernal.view(4* self.k, 256, 7, 7)
            rinput  = self.conv4(detection)

            coutput = F.conv2d(cinput, ckernal)
            routput = F.conv2d(rinput, rkernal)
            """
            print('++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('c branch conv1 template  weight', self.conv1.weight[0,0,0,0])
            print('c branch conv3 detection weight', self.conv3.weight[0,0,0,0])
            """
            return coutput, routput
    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        print('Resume checkpoint')

class RDN(nn.Module):
    def __init__(self):
        super(RDN,self).__init__()
        self.conv1 = nn.Conv2d(3, 72, kernel_size=7, stride=2, padding=0, bias=False)

        self.dense_block1 = DenseBlock(72,36,2)
        self.transition1 = transitionlayer(self.dense_block1.out_channels,36)
        self.dense_block2 = DenseBlock(36,36,4)
        self.transition2 = transitionlayer(self.dense_block2.out_channels,36)
        self.dense_block3 = DenseBlock(36,36,6)
        self.dense_block4 = DenseBlock_b(in_channels=252,kernel_size=7)
    def forward(self,x):
        x = self.conv1(x)

        x = self.dense_block1(x)
        x = self.transition1(x)

        x = self.dense_block2(x)
        x = self.transition2(x)

        x = self.dense_block3(x)
        x = self.dense_block4(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, concat_input=True):
        super(DenseBlock,self).__init__()
        self.concat_input = concat_input
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.out_channels = num_layers * growth_rate
        if self.concat_input:
            self.out_channels += self.in_channels
        for i in range(num_layers):
            self.add_module(f'layer_{i}',
                            DenseLayer(in_channels=in_channels+i*growth_rate,out_channels = growth_rate))

    def forward(self,block_input):
        layer_input = block_input
        layer_output = block_input.new_empty(0)
        all_outputs = [block_input] if self.concat_input else []
        for layer in self._modules.values():
            layer_input = torch.cat([layer_input, layer_output], dim=1)
            layer_output = layer(layer_input)
            all_outputs.append(layer_output)

        return torch.cat(all_outputs, dim=1)

class DenseBlock_b(nn.Sequential):
    def __init__(self, in_channels, kernel_size):
        super(DenseBlock_b,self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.add_module('bn1',nn.BatchNorm2d(num_features=self.in_channels))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(self.in_channels,out_channels=384,kernel_size=self.kernel_size))
        self.add_module('dropout',nn.Dropout2d(0.2,inplace=True))

        self.add_module('bn1',nn.BatchNorm2d(num_features=self.in_channels))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(self.in_channels,out_channels=384,kernel_size=self.kernel_size))
        self.add_module('dropout',nn.Dropout2d(0.2,inplace=True))

        self.add_module('bn1',nn.BatchNorm2d(num_features=self.in_channels))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(self.in_channels,out_channels=256,kernel_size=self.kernel_size))
        self.add_module('dropout',nn.Dropout2d(0.2,inplace=True))

class DenseLayer(nn.Sequential):
    r"""
    Consists of:
    - Batch Normalization
    - ReLU
    - 3x3 Convolution
    - (Dropout) dropout rate: 0.2
    """

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DenseLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
        if dropout > 0:
            self.add_module('drop', nn.Dropout2d(dropout, inplace=True))

class transitionlayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(transitionlayer,self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=1)
        self.average = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.average(x)
        return x

class SiameseRPN(nn.Module):
    def __init__(self):
        super(SiameseRPN, self).__init__()
        self.features = nn.Sequential(                  #1, 3, 256, 256
            nn.Conv2d(3, 64, kernel_size=11, stride=2), #1, 64,123, 123
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1, 64, 60,  60
            nn.Conv2d(64, 192, kernel_size=5),          #1,192, 56,  56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1,192, 27,  27
            nn.Conv2d(192, 384, kernel_size=3),         #1,384, 25,  25
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),         #1,256, 23,  23
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),         #1,256, 21,  21
        )
        self.k = 5
        self.s = 4
        self.conv1 = nn.Conv2d(256, 2*self.k*256, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4*self.k*256, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(256, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(256, 4* self.k, kernel_size = 4, bias = False)

    def forward(self, template, detection):
        template = self.features(template)
        detection = self.features(detection)

        ckernal = self.conv1(template)
        ckernal = ckernal.view(2* self.k, 256, 4, 4)
        cinput  = self.conv3(detection)

        rkernal = self.conv2(template)
        rkernal = rkernal.view(4* self.k, 256, 4, 4)
        rinput  = self.conv4(detection)

        coutput = F.conv2d(cinput, ckernal)
        routput = F.conv2d(rinput, rkernal)
        """
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('c branch conv1 template  weight', self.conv1.weight[0,0,0,0])
        print('c branch conv3 detection weight', self.conv3.weight[0,0,0,0])
        """
        return coutput, routput

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Load Alexnet models Done' )

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        print('Resume checkpoint')


class SiameseRPN_bn(nn.Module):
    def __init__(self):
        super(SiameseRPN_bn, self).__init__()
        self.features = nn.Sequential(                  #1, 3, 256, 256
            nn.Conv2d(3, 64, kernel_size=11, stride=2), #1, 64,123, 123
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1, 64, 60,  60
            nn.Conv2d(64, 192, kernel_size=5),          #1,192, 56,  56
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1,192, 27,  27
            nn.Conv2d(192, 384, kernel_size=3),         #1,384, 25,  25
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),         #1,256, 23,  23
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),         #1,256, 21,  21
            nn.BatchNorm2d(256),
        )

        self.k = 5
        self.s = 4
        self.conv1 = nn.Conv2d(256, 2*self.k*256, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(2*self.k*256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4*self.k*256, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(4*self.k*256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn3   = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn4   = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(256, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(256, 4* self.k, kernel_size = 4, bias = False)

        #self.reset_params()

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Load Alexnet models Done' )

    def forward(self, template, detection):
        """
        把template的类别,坐标的特征作为检测cconv和rconv的检测器
        把ckernel, rkernel转换到cconv, rconv
        """
        template = self.features(template)            #
        detection = self.features(detection)          #21

        ckernal = self.bn1(self.conv1(template))
        ckernal = ckernal.view(2* self.k, 256, 4, 4)
        cinput  = self.bn3(self.conv3(detection))                #21 -> 19


        rkernal = self.bn2(self.conv2(template))
        rkernal = rkernal.view(4* self.k, 256, 4, 4)
        rinput  = self.bn4(self.conv4(detection))

        coutput = F.conv2d(cinput, ckernal)
        routput = F.conv2d(rinput, rkernal)
        """
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('c branch conv1 template  weight', self.conv1.weight[0,0,0,0])
        print('c branch conv3 detection weight', self.conv3.weight[0,0,0,0])
        """
        return coutput, routput

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        print('Resume checkpoint')

if __name__ == '__main__':
    model = SiameseRPN()

    template = torch.ones((1, 3, 127, 127))
    detection= torch.ones((1, 3, 256, 256))

    y1, y2 = model(template, detection)
    print(y1.shape) #[1, 10, 17, 17]
    print(y2.shape) #[1, 20, 17, 17]15
