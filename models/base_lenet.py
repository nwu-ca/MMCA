import torch.nn as nn
import torch.nn.functional as F

# 类，继承nn.Module
class LeNet(nn.Module):
    # 初始化函数
    def __init__(self):
        super(LeNet, self).__init__()  # 使用多继承都会使用super
        self.conv1 = nn.Conv2d(3, 16, 5)  # in_channels,out_channels,kernel_size
        self.pool1 = nn.MaxPool2d(2, 2)  #kernel_size,stride
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)  # 全连接层需要展平，变成1维向量
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 第二个参数，根据数据集的类别确定，这里使用cifar-10

    # 定义正向传播, x表示输入的数据
    def forward(self, x):            # input(3, 32, 32)         N = (W-F+2P)/S + 1
        x = F.relu(self.conv1(x))    # 卷积 output(16, 28, 28)  28 = (32-5+2*0)/1 + 1
        x = self.pool1(x)            # 池化 output(16, 14, 14)
        x = F.relu(self.conv2(x))    # 卷积 output(32, 10, 10)  10 = (14-5+2*0)/1 + 1
        x = self.pool2(x)            # 池化 output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # 全连接 output (120)
        x = F.relu(self.fc2(x))      # 全连接 output(84)
        x = self.fc3(x)              # 全连接 output(10)
        return x
