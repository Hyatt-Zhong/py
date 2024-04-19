import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设定超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 13

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载并加载数据
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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

model = Autoencoder()
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train(model, dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

train(model, train_loader, num_epochs)

torch.save(model, 'autoencode.pth')