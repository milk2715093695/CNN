import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

class CNN_MNIST(nn.Module):
    def __init__(self):
        # 配置日志
        self.logger = logging.getLogger()

        super(CNN_MNIST, self).__init__()
        
        # 第一层卷积，输入1通道，输出32通道，卷积核大小3，步长1，padding1，图像大小不变
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积，输入32通道，输出64通道，卷积核大小3，步长1，padding1，图像大小不变
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 第三层卷积，输入64通道，输出128通道，卷积核大小3，步长1，padding1，图像大小不变
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 将28x28的图像展平后连接到全连接层
        self.fc2 = nn.Linear(512, 10)  # 10个类别输出
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor):
        self.logger.debug("开始预测\n")
        self.logger.debug(f"输入维度：{x.shape}\n")
        # 第一层卷积 + 激活 + 最大池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 2x2的池化
        self.logger.debug(f"第一个卷积层输出维度：{x.shape}\n")

        # 第二层卷积 + 激活 + 最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        self.logger.debug(f"第二个卷积层输出维度：{x.shape}\n")

        # 第三层卷积 + 激活 + 最大池化
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        self.logger.debug(f"第三个卷积层输出维度：{x.shape}\n")

        # 展平
        x = x.view(-1, 128 * 3 * 3)
        self.logger.debug(f"展平后维度：{x.shape}\n")

        # 全连接层 + 激活 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        self.logger.debug(f"第一个全连接层输出维度：{x.shape}\n")

        # 输出层
        x = self.fc2(x)
        self.logger.debug(f"网络的输出维度：{x.shape}\n")
        return x