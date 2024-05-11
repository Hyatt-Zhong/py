import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

A = torch.tensor([[1, 2], [3, 4]])
C = torch.tensor([[5, 6], [7, 8]])

# 逐元素相乘
result = A * C

print(result)


# class MeIsMe(torch.nn.Module):
#     def __init__(self):
#         super(MeIsMe, self).__init__()
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 10, kernel_size=5),
#             # torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2),
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.Conv2d(10, 20, kernel_size=5),
#             # torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2),
#         )
#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(320, 50),
#             torch.nn.Linear(50, 10),
#         )

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.conv1(x)  
#         x = self.conv2(x)  
#         x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
#         x = self.fc(x)
#         return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
    
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.fc = nn.Linear(10,10)

#     def forward(self, x):
#         x = self.fc(x)
#         return x
    

# # 数据转换
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# batch_size = 64

# # 下载并加载数据
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# encodec = torch.load('encodec.pth')
# model = torch.load('meisme.pth')

