import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image
from torchvision import transforms as tfs
from matplotlib import pyplot as plt
import datetime

# def get_upsampling_weight(in_channels, out_channels, kernel_size):
#     """Make a 2D bilinear kernel suitable for upsampling"""
#     factor = (kernel_size + 1) // 2
#     if kernel_size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:kernel_size, :kernel_size]
#     filt = (1 - abs(og[0] - center) / factor) * \
#            (1 - abs(og[1] - center) / factor)
#     weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
#                       dtype=np.float64)
#     weight[range(in_channels), range(out_channels), :, :] = filt
#     return torch.from_numpy(weight).float()

# class FCN8s(nn.Module):

#     def __init__(self, numclass):
#         super(FCN8s, self).__init__()
#         #(320, 480)
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         #(160, 240)
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         #(80, 120)
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         #(40, 60)
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         # (20, 30)
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         # (10, 15)
#         self.conv_1 = nn.Conv2d(in_channels=256, out_channels=numclass, kernel_size=3, padding=1)
#         self.conv_2 = nn.Conv2d(in_channels=512, out_channels=numclass, kernel_size=3, padding=1)
#         self.tranConv2x = nn.ConvTranspose2d(in_channels=numclass, out_channels=numclass, kernel_size=4, stride=2, padding=1)
#         self.tranConv8x = nn.ConvTranspose2d(in_channels=numclass, out_channels=numclass, kernel_size=16, stride=8, padding=4)

#         self._initialize_weights()

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x1 = self.layer3(x)       #(40, 60), channel = 256
#         x2 = self.layer4(x1)      #(20, 30), channel = 512
#         x3 = self.layer5(x2)      #(10, 15), channel = 512
        
#         x1 = self.conv_1(x1)      #(40, 60), channel = numclass
#         x2 = self.conv_2(x2)      #(20, 30), channel = numclass
#         x3 = self.conv_2(x3)      #(10, 15), channel = numclass

#         x3 = self.tranConv2x(x3)  #(20, 30), channel = numclass
#         x2 += x3                  #(20, 30), channel = numclass
#         x2 = self.tranConv2x(x2)  #(40, 60), channel = numclass
#         x1 += x2                  #(40, 60), channel = numclass
#         x1 = self.tranConv8x(x1)  #(320, 480), channel = numclass
#         return x1
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.zero_()
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             if isinstance(m, nn.ConvTranspose2d):
#                 assert m.kernel_size[0] == m.kernel_size[1]
#                 initial_weight = get_upsampling_weight(
#                     m.in_channels, m.out_channels, m.kernel_size[0])
#                 m.weight.data.copy_(initial_weight)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.pretrained_net = torchvision.models.resnet.resnet34(pretrained=True)
        self.stage1 = nn.Sequential(*list(self.pretrained_net.children())[:-4]) # 第一段
        self.stage2 = list(self.pretrained_net.children())[-4] # 第二段
        self.stage3 = list(self.pretrained_net.children())[-3] # 第三段
        
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) # 使用双线性 kernel
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel

        
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        # s2 = self.upsample_4x(s2)
        s2 = self.upsample_2x(s2)
        s = s1 + s2

        s = self.upsample_8x(s)
        return s
