'''transforms模块提供常用的图像变换，张量变换操作。这些变换可以通过 Compose 进行链式组合。'''
import random
from email.charset import add_alias
from email.headerregistry import Address

import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
'''自定义transforms，要实现__call__方法'''
class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

#python -m tensorboard.main --logdir=log,可视化查看内容
#只有numpy和张量可以放入tensorboard中
#图像的transforms之后要进行张量化处理
img = Image.open("img/test.jpg")
print(img)
writer = SummaryWriter("log")
img_center_crop = transforms.CenterCrop(100) # 中心裁剪，先实现这个工具
img_trans = img_center_crop(img)  #再利用这个工具对图像进行处理
img_to_tensor = transforms.ToTensor()  # 同样先实现工具
img_trans = img_to_tensor(img_trans)  #再利用工具
writer.add_image("tensor1", img_trans, global_step=1)



img2 = Image.open("img/175998972.jpg")
#多个transforms可以通过Compose进行组合，直接将工具放入Compose中，不用一个个去实现
addpepper = AddPepperNoise(snr=0.5, p=0.9)
img_transforms = transforms.Compose([
    transforms.CenterCrop(100),  # 中心裁剪
    addpepper,  # 添加椒盐噪声
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
])
img_trans_compose=img_transforms(img2)
writer.add_image("tensor2", img_trans_compose,global_step=2)
writer.close()

