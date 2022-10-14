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


def read_images(root='.\\data\\VOCdevkit\\VOC2012', train=True):
    txt_fname = root + '\\ImageSets\\Segmentation\\' + \
        ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label


class MyDataSet(Dataset):
    def __init__(self, train, crop_size, transforms):
        super(MyDataSet, self).__init__()
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)

    def _filter(self, images):  # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        data = Image.open(data)
        label = Image.open(label).convert('RGB')
        input_shape = (320, 480)
        data, label = self.transforms(data, label, input_shape)
        # print(label.shape) (320,480)
        return data, label

    def __len__(self):
        return len(self.data_list)


classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i  # 建立索引


def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵


def rand_crop(data, label, height, width):
    '''
    data is PIL.Image object
    label is PIL.Image object
    '''
    i, j, h, w = tfs.RandomCrop.get_params(data, output_size=(height, width))
    data = TF.crop(data, i, j, h, w)
    label = TF.crop(label, i, j, h, w)
    return data, label

input_shape = (320, 480)

def img_transforms(im, label, crop_size):
    im, label = rand_crop(im, label, *crop_size)
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    im = im_tfs(im)

    label = image2label(label)
    label = torch.from_numpy(label)
    return im, label


train_set = MyDataSet(True, input_shape, transforms=img_transforms)
test_set = MyDataSet(False, input_shape, transforms=img_transforms)

# x, y = Image.open(train_set.data_list[0]), Image.open(train_set.label_list[0]).convert('RGB')
# x, y = img_transforms(x, y)
# x = torch.Tensor.squeeze(x, dim=1)
# print(x.shape , y.shape)

class ScheduledOptim(object):
    '''A wrapper class for learning rate scheduling'''
 
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']   #长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数；
        self.current_steps = 0
 
    def step(self):
        "Step by the inner optimizer"
        self.current_steps += 1
        self.optimizer.step()
 
    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()
 
    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
 
    @property
    def learning_rate(self):
        return self.lr

train_data = DataLoader(train_set, shuffle=True, batch_size=64)
test_data = DataLoader(test_set, shuffle=True, batch_size=64)

model = FCN8s(21)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=1e-4)
optimizer = ScheduledOptim(optimizer)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epoch_nums = 10
loss_list = []
epoch_list = []

for x in range(epoch_nums):
    i = 0
    epochLoss = 0
    print('第{}个epoch.'.format(x + 1))
    for imgs, labels in train_data:
        imgs = imgs.to(device)
        labels = labels.to(device)
        #print(imgs.shape)     # torch.Size([64, 3, 320, 480])
        # print(labels.shape)  # torch.Size([64, 320, 480])
        print('第{}个epoch第{}个mini-batch.'.format(x + 1, i + 1))
        outputs = model(imgs)
        #print(outputs.shape)  # torch.Size([64, 21, 320, 480])
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('第{}个epoch第{}个mini-batch结束.'.format(x + 1, i + 1))
        epochLoss += loss.item()
        i += 1
    loss_list.append(epochLoss)
    epoch_list.append(x)
    print('第{}个epoch结束.'.format(x + 1))

torch.save(model, "FCN.nn")

plt.plot(epoch_list, loss_list)
plt.xlabel('dencent times')
plt.ylabel('loss')
plt.show()
