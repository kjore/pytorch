

#构建神经网络模型

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

#这段代码只有在直接运行model.py文件时才会执行，而不会在导入该模块时执行
if __name__ == '__main__':
    Mynn = Mynn()
    input = torch.ones((64, 3, 32, 32))
    output = Mynn(input)
    print(output.shape)
    writer = SummaryWriter("log_Model")
    writer.add_graph(Mynn, input)
    writer.close()