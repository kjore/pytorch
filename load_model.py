import torch
from model import Mynn

# 第一种保存方式对应的加载方式
model1 = torch.load("mynn_method1.pth", weights_only=False)
print(model1)

# 第二种保存方式对应的加载方式
model2 = Mynn()
model2.load_state_dict(torch.load("mynn_method2.pth", weights_only=False))
print(model2)