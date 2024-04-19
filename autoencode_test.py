import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

# 定义自编码器的架构
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 输出: [16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 输出: [32, 7, 7]
            nn.ReLU(),
            nn.Flatten(),  # 展平
            nn.Linear(32 * 7 * 7, 64),  # 输出: 64
            nn.ReLU(),
            nn.Linear(64, 16),  # 输出: 16
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32 * 7 * 7),
            nn.ReLU(),
            nn.Linear(32 * 7 * 7, 784),
            nn.ReLU(),
            nn.Unflatten(1,(1, 28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

model = torch.load('autoencode.pth')

# for data in train_loader:
#     img, _ = data
#     print(img.shape)
#     break

def read_gray_image(pic):
    image = cv2.imread(pic)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

img = read_gray_image('image28.jpg')
img = torch.from_numpy(img).float()
img = img.unsqueeze(0).unsqueeze(0)
print(img.shape)

outputs = model(img)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))

plt.imshow(outputs.view(28, 28).detach().numpy(), cmap='gray')
plt.show()

# 使用训练好的模型重建图像
# with torch.no_grad():
#     for data in train_loader:
#         img, _ = data
#         # img = img.view(img.size(0), -1)
#         outputs = model(img)
#         break

# # 显示原始图像和重建图像
# import matplotlib.pyplot as plt
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # 原始图像
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(img[i].view(28, 28).numpy(), cmap='gray')
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # 重建图像
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(outputs[i].view(28, 28).numpy(), cmap='gray')
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()