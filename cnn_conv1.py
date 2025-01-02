import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的一维卷积网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(32 * 50, 10)  # 假设输入信号长度为100

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 通过第一个卷积层和池化层
        x = self.pool(torch.relu(self.conv2(x)))  # 通过第二个卷积层和池化层
        x = x.view(-1, 32 * 50)  # 将特征展平为一维向量
        x = self.fc(x)  # 全连接层
        return x

# 创建模型
model = SimpleCNN()

# 生成一个随机的信号 (batch_size, in_channels, signal_length)
signal = torch.randn(8, 1, 100)  # 8个样本，每个样本是1维长度为100的信号

# 前向传播
output = model(signal)
print(output.shape)  # 输出的形状为 (batch_size, 10)