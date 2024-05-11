import torch

# 创建两个向量
x = torch.tensor([1, 1], dtype=torch.float32)
y = torch.tensor([2, 2], dtype=torch.float32)

# 计算欧氏距离
distance = torch.dist(x, y)
print(distance)  # 输出欧氏距离
manhattan_distance = torch.sum(torch.abs(x - y))
print(manhattan_distance)  # 输出曼哈顿距离