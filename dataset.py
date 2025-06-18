import torch
import torchvision
from torch import nn
'''从现有数据集中加载数据集'''
train_data = torchvision.datasets.CIFAR10(root="./data",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)#训练集
test_data = torchvision.datasets.CIFAR10(root="./data",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)#测试集

'''自定义数据集'''
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data # data是一个列表，包含图像和标签的元组,对于不同的文件（.txt .csv），获取data方法不同
        self.transform = transform
        self.target_transform = target_transform
#transform用于对图像进行预处理，target_transform用于对标签进行预处理

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

'''还可以自定义transfrom'''
class MyTransform:
    def __call__(self, img):
        # 在这里定义你的转换逻辑
        return img

dataset = MyDataset(train_data, transform=MyTransform())