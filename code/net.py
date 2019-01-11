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

#    def temple(self,z):
#       z_f = self.features(z)
 #       rkernel_raw = self.conv1(z_f)
#        ckernel_raw = self.conv3(z_f)
 #       kernel_size = rkernel_raw.data.size()[-1]
 #       self.rkernel = rkernel_raw.view(self.k*4,self.s,3,3)
 #       self.ckernel = ckernel_raw.view(self.k*2,self.s,3,3)

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
