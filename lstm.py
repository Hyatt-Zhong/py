# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# # 生成正弦波数据
# def generate_sine_wave(seq_length):
#     x = np.linspace(0, 50, seq_length + 1)
#     return np.sin(x)

# # 定义 LSTM 模型
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out

# # 参数设置
# input_size = 1
# hidden_size = 50
# num_layers = 1
# output_size = 1
# num_epochs = 300
# learning_rate = 0.01
# seq_length = 50

# # 创建模型
# model = LSTMModel(input_size, hidden_size, num_layers, output_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # 生成数据
# data = generate_sine_wave(seq_length + 10)
# input_seq = torch.tensor(data[:-1]).float().view(-1, seq_length, input_size)
# target_seq = torch.tensor(data[1:]).float().view(-1, seq_length, output_size)

# # 训练模型
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     output = model(input_seq)
#     loss = criterion(output, target_seq[:, -1, :])
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 30 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # 测试模型
# model.eval()
# predicted = model(input_seq).detach().numpy()

# # 绘制结果
# plt.plot(data, label='Original data')
# plt.plot(np.arange(seq_length, seq_length + len(predicted)), predicted, label='Predicted')
# plt.legend()
# plt.show()









import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义 LSTM 单元
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weights_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = (torch.mm(input, self.weights_ih.t()) + torch.mm(hx, self.weights_hh.t()) + self.bias)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

# 生成正弦波数据
def generate_sine_wave(seq_length):
    x = np.linspace(0, 50, seq_length + 1)
    return np.sin(x)

# 参数设置
input_size = 1
hidden_size = 20
seq_length = 50
num_epochs = 300
learning_rate = 0.01

# 创建LSTM单元和优化器
lstm = LSTMCell(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# 生成数据
data = generate_sine_wave(seq_length + 1)
inputs = torch.tensor(data[:-1]).float().view(-1, 1, input_size)
targets = torch.tensor(data[1:]).float().view(-1, 1, input_size)

# 训练模型
for epoch in range(num_epochs):
    hx = torch.zeros(1, hidden_size)
    cx = torch.zeros(1, hidden_size)
    loss = 0
    for i in range(seq_length):
        input = inputs[i]
        target = targets[i]
        hx, cx = lstm(input, (hx, cx))
        loss += criterion(hx, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 30 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()/seq_length:.4f}')

# 测试模型
predicted = []
hx = torch.zeros(1, hidden_size)
cx = torch.zeros(1, hidden_size)
for i in range(seq_length):
    input = inputs[i]
    hx, cx = lstm(input, (hx, cx))
    predicted.append(hx.detach().numpy().ravel()[0])

# 绘制结果
plt.plot(data, label='Original data')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()


# __init__ 方法
# def __init__(self, input_size, hidden_size):
#     super(LSTMCell, self).__init__()
#     self.input_size = input_size
#     self.hidden_size = hidden_size
#     self.weights_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
#     self.weights_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
#     self.bias = nn.Parameter(torch.randn(4 * hidden_size))

# 输入参数:

# input_size: 输入数据的维度。
# hidden_size: 隐藏状态（hidden state）的维度。
# 初始化父类:

# super(LSTMCell, self).__init__() 调用父类 nn.Module 的初始化方法。
# 定义权重参数和偏置:

# self.weights_ih: 输入到隐藏状态的权重矩阵，形状为 (4 * hidden_size, input_size)。这里的 4 * hidden_size 是因为LSTM有四个门（输入门、遗忘门、细胞状态更新和输出门）。
# self.weights_hh: 隐藏状态到隐藏状态的权重矩阵，形状为 (4 * hidden_size, hidden_size)。
# self.bias: 偏置向量，形状为 (4 * hidden_size)。
# 这些权重和偏置在训练过程中会被优化。

# forward 方法
# def forward(self, input, hidden):
#     hx, cx = hidden
#     gates = (torch.mm(input, self.weights_ih.t()) + torch.mm(hx, self.weights_hh.t()) + self.bias)
#     ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

#     ingate = torch.sigmoid(ingate)
#     forgetgate = torch.sigmoid(forgetgate)
#     cellgate = torch.tanh(cellgate)
#     outgate = torch.sigmoid(outgate)

#     cy = (forgetgate * cx) + (ingate * cellgate)
#     hy = outgate * torch.tanh(cy)

#     return hy, cy

# 输入参数:

# input: 当前时间步的输入，形状为 (batch_size, input_size)。
# hidden: 一个元组，包含前一个时间步的隐藏状态 hx 和细胞状态 cx。
# 分解隐藏状态:

# hx, cx = hidden 将隐藏状态和细胞状态解包。
# 计算门控值:

# gates = (torch.mm(input, self.weights_ih.t()) + torch.mm(hx, self.weights_hh.t()) + self.bias):
# torch.mm(input, self.weights_ih.t()) 表示将输入与输入到隐藏状态的权重矩阵相乘。
# torch.mm(hx, self.weights_hh.t()) 表示将前一个隐藏状态与隐藏状态到隐藏状态的权重矩阵相乘。
# 加上偏置 self.bias。
# gates.chunk(4, 1): 将计算结果分成四个部分，每个部分对应一个门的值。
# 激活函数:

# ingate = torch.sigmoid(ingate): 输入门的值通过 sigmoid 激活函数。
# forgetgate = torch.sigmoid(forgetgate): 遗忘门的值通过 sigmoid 激活函数。
# cellgate = torch.tanh(cellgate): 细胞状态更新的值通过 tanh 激活函数。
# outgate = torch.sigmoid(outgate): 输出门的值通过 sigmoid 激活函数。
# 更新细胞状态和隐藏状态:

# cy = (forgetgate * cx) + (ingate * cellgate): 更新细胞状态。细胞状态是前一个细胞状态 cx 经过遗忘门 forgetgate 的部分加上新的细胞状态 cellgate 经过输入门 ingate 的部分。
# hy = outgate * torch.tanh(cy): 更新隐藏状态。隐藏状态是新的细胞状态 cy 经过 tanh 激活函数后，再乘以输出门 outgate 的值。
# 返回新的隐藏状态和细胞状态:

# return hy, cy: 返回新的隐藏状态和细胞状态。