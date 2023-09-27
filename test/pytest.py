

import torch

# 创建一个张量
x = torch.tensor([1, 6, 3, 8, 2, 4])
y = torch.tensor([1, 6, 3, 8, 2, 4])

# 使用逻辑运算符和索引操作来对元素赋值
x[y > 5] = 0

print(x)