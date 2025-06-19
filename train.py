import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
from model import*


#准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data",train=True,transform=transforms.ToTensor(),download=True)  # 训练集
#准备测试集
test_data = torchvision.datasets.CIFAR10(root="./data",train=False,transform=transforms.ToTensor(),download=True)  # 测试集

train_len = len(train_data)
test_len = len(test_data)
print(f"训练集长度: {train_len}, 测试集长度: {test_len}")

# 准备数据加载器
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

# 准备模型
device = "cuda" if torch.cuda.is_available() else "cpu"  # 检查是否有GPU可用，如果有则使用GPU
mynn = Mynn().to(device)  # 检查是否有GPU可用，如果有则使用GPU

# 准备损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mynn.parameters(), lr=0.001)

#训练参数设置
total_train_step = 0
total_test_step = 0
epoch = 10   #训练轮数

# 准备TensorBoard
writer = SummaryWriter("logMynn")

#开始训练
for i in range(epoch):
    print(f"-------第{i+1}轮训练开始-------\n")

    # 训练步骤开始
    mynn.train()
    for data in train_loader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到GPU上

        '''首先清空梯度
        然后进行前向计算
        计算损失
        反向传播计算梯度
        最后更新参数'''
        # 优化器优化模型
        optimizer.zero_grad()
        outputs = mynn(imgs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {loss.item()}\n")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    mynn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到GPU上
            outputs = mynn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_len))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_len, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(mynn, "mynni/mynn_{}.pth".format(i))
    print("模型已保存")

writer.close()