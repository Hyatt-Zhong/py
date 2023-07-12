import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

# 1. 定义数据
x = torch.rand([50, 2])
y = x[:, 0] * 6 + x[:, 1] * 9 + 40


# 2 .定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入特征维度为2，输出特征维度为1

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

num_epochs = 100
for epoch in range(num_epochs):
    inputs = x
    labels = y

    optimizer.zero_grad()

    outputs = model(inputs)
    print(inputs)
    loss = criterion(outputs.squeeze(), labels)#这里和单权重的线性回归不一样

    loss.backward()
    optimizer.step()

for param in model.parameters():
    print(param)
