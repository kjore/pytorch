import torch
import torchvision
from torch import nn
from model import Mynn

mynn = Mynn()

#保存方式1  模型结构+模型参数，较大
torch.save(mynn,"mynn_method1.pth")

#保存方式2  模型参数（官方推荐）
torch.save(mynn.state_dict(),"mynn_method2.pth")

