import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MeIsMe(torch.nn.Module):
    def __init__(self):
        super(MeIsMe, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Linear(10,10)

    def forward(self, x):
        x = self.fc(x)
        return x
    

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

encodec = MeIsMe()
model = Net()
# 交叉熵损失和均方差损失维度不同
# criterion = nn.MSELoss()  # 使用均方误差损失
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(encodec, model, dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data in dataloader:
            img, target = data
            # print(img.shape)
            encode = encodec(img)
            output = model(encode)
            # print(output.shape, target.shape, target)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

train(encodec, model, train_loader, num_epochs)

torch.save(encodec, 'encodec.pth')
torch.save(model, 'meisme.pth')