from select import select
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
from FCNModel import FCN8s
import torchvision.transforms.functional as TF
import random

def pick_random_imgs(cropSize, num, root='.\\data\\VOCdevkit\\VOC2012'):
    txt_fname = root + '\\ImageSets\\Segmentation\\val.txt'
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    dataList = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    labelList = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]

    size = len(dataList)
    selectedData = []
    selectedLabel = []

    while num > 0:
        r = random.randint(0, size)
        im = dataList[r]
        if (selectedData.count(im) == 0 and Image.open(im).size[1] >= cropSize[0]) and (Image.open(im).size[0] >= cropSize[1]):
            selectedData.append(dataList[r])
            selectedLabel.append(labelList[r])
            num -= 1
    
    croppedData = []
    croppedLabel = []
    for data, label in zip(selectedData, selectedLabel):
        d = Image.open(data)
        l = Image.open(label).convert('RGB')
        d, l = rand_crop(d, l, *cropSize)
        croppedData.append(d)
        croppedLabel.append(l)
    
    return croppedData, croppedLabel

def rand_crop(data, label, height, width):
    '''
    data is PIL.Image object
    label is PIL.Image object
    '''
    i, j, h, w = tfs.RandomCrop.get_params(data, output_size=(height, width))
    data = TF.crop(data, i, j, h, w)
    label = TF.crop(label, i, j, h, w)
    return data, label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("FCN.nn")
model = model.to(device)

def img_transforms(img):
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformedImg = im_tfs(img)
    return transformedImg

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

pickNums = 5
d, l = pick_random_imgs((320, 480), pickNums)
p = []
cm = np.array(colormap).astype('uint8')
for img in d:
    img = img_transforms(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)
    out = model(img)
    pred = out.max(1)[1].squeeze().data.numpy()
    pred = cm[pred] 
    p.append(pred)

i = 1
for x, y, z in zip(d, l, p):
    plt.subplot(pickNums, 3, i)
    plt.imshow(x)
    i += 1
    plt.subplot(pickNums, 3, i)
    plt.imshow(y)
    i += 1
    plt.subplot(pickNums, 3, i)
    plt.imshow(z)
    i += 1

plt.show()

# input_shape = (320, 480)
# img = img_transforms(img, input_shape)
# img = torch.unsqueeze(img, dim=0)
# out = model(img)
# print(out.shape)  #torch.Size([1, 21, 320, 480])
# cm = np.array(colormap).astype('uint8')
# pred = out.max(1)[1].squeeze().data.numpy()
# print(pred.shape)  #(320, 480)
# pred = cm[pred] 
# print(pred.shape)
# plt.subplot(1, 2, 2)
# plt.imshow(pred)
# plt.show()

