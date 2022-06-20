# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:11:06 2022

@author: Admin
"""

import torch.nn as nn
import torch
from torchvision import models

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0, in_channel = 320):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(in_channel, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.intermediate = BasicConv2d(512, 320, kernel_size=3, padding=1, stride=1)
        # self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.backend = Block35(in_channel=320)
        # self.backend_repeat = Block35(in_channel=320)
        # self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.output_layer = BasicConv2d(320, 1, kernel_size=3, padding=1, stride=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        x = self.intermediate(x)
        x = self.backend(x)
        x = self.backend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)               

def printModelParams(file):
    from pytorch_model_summary import summary
    from PIL import Image
    from torch.autograd import Variable
    from torchvision import transforms
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
    img = Image.open(file).convert('RGB')
    img = transform(img)
    # the below code is needed to insert singleton batch because pytorch expects a batch of images as an input.
    img = img[None, ...]
    model = CSRNet().cpu()
    print(summary(model, Variable(img.cpu())))

def printModel():
    path = "C:\\Users\\Admin\\Desktop\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\images\\img_0008.jpg"
    # path = "C:\\Users\\Admin\\Desktop\\TA\\Dataset\\mall_dataset\\frames\\seq_000001.jpg"
    printModelParams(path)