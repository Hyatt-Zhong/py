import torch.nn as nn
import torch


im = torch.randn(1, 1, 5, 5)
im = torch.tensor(
    [
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    ],
    dtype=torch.float32,
)
c = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0)
c.weight.data = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
c.bias.data.fill_(0)
output = c(im)

# print(im)
print(output)
# print(list(c.parameters()))


class net(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(net, self).__init__()

        self.c = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0)
        self.c.weight.data = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        self.c.bias.data.fill_(0)

    def forward(self, x):
        out = self.c(x)
        return out


mynet = net()
y = mynet(im)

print(y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）

# 首先通过一个卷积核为5×5的卷积层，其通道数从1变为10，高宽分别为24像素；
# 然后通过一个卷积核为2×2的最大池化层，通道数不变，高宽变为一半，即维度变成（batch,10,12,12）；
# 然后再通过一个卷积核为5×5的卷积层，其通道数从10变为20，高宽分别为8像素；
# 再通过一个卷积核为2×2的最大池化层，通道数不变，高宽变为一半，即维度变成（batch,20,4,4）；
# 之后将其view展平，使其维度变为320(2044)之后进入全连接层，用线性函数将其输出为10类，即“0-9”10个数字。

