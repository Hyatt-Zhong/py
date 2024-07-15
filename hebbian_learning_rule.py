import torch
import torch.nn as nn
import torch.optim as optim

####多次刺激加深记忆######基于神经元之间的联结强度的调整

# # 定义一个简单的神经网络
# class SimpleHebbianNet(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(SimpleHebbianNet, self).__init__()
#         self.fc = nn.Linear(input_size, output_size, bias=False)

#     def forward(self, x):
#         return self.fc(x)

# # Hebbian学习规则的实现
# def hebbian_learning_rule(weight, input, output, learning_rate):
#     delta_w = learning_rate * torch.outer(output, input)
#     weight.data += delta_w

# # 超参数
# input_size = 3
# output_size = 2
# learning_rate = 0.01
# epochs = 100



# # 创建数据
# # 假设我们有一些简单的输入数据和目标输出
# inputs = torch.tensor([[1.0, 0.5, -1.0],
#                        [-0.5, -1.0, 0.5],
#                        [1.0, 1.0, 1.0],
#                        [-1.0, -0.5, 0.5]], dtype=torch.float32)

# # 创建神经网络实例
# model = SimpleHebbianNet(input_size, output_size)

# x = torch.tensor([0.5, -0.5, 0.5], dtype=torch.float32)
# y = model(x)
# print(f'Test Input: {x}, Test Output: {y}')

# # 训练过程
# for epoch in range(epochs):
#     for input in inputs:
#         # 前向传播
#         output = model(input)

#         # 计算Hebbian学习规则的权重更新
#         hebbian_learning_rule(model.fc.weight, input, output, learning_rate)

#     # 打印训练进度
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Weights: \n{model.fc.weight.data}')

# # 测试
# test_input = torch.tensor([0.5, -0.5, 0.5], dtype=torch.float32)
# test_output = model(test_input)
# print(f'Test Input: {test_input}, Test Output: {test_output}')


import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的单层前馈神经网络
class HebbianNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(HebbianNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.fc(x)

    def hebbian_update(self, x, y, lr=0.01):
        delta_w = lr * torch.matmul(y.t(), x)  # Hebbian学习规则
        self.fc.weight.data += delta_w  # 更新权重
        self.normalize_weights()  # 归一化权重

    def normalize_weights(self):
        # 使用L2归一化来限制权重的增长
        with torch.no_grad():
            norm = self.fc.weight.norm(p=2, dim=1, keepdim=True)
            self.fc.weight.div_(norm)

# 生成一些随机数据
input_size = 10
output_size = 5
batch_size = 10
num_epochs = 100

x = torch.randn(batch_size, input_size)
y = torch.randn(batch_size, output_size)

# 实例化网络
net = HebbianNetwork(input_size, output_size)

# 训练网络
for epoch in range(num_epochs):
    output = net(x)
    net.hebbian_update(x, output, lr=0.01)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Weights: {net.fc.weight}")

print("Training complete!")