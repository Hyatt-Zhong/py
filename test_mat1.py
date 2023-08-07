import matrix as mat
import random

import torch.nn as nn
import torch

# pip3 install torch torchvision torchaudio
# 2.0.1+cpu

# print(torch.__version__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        conv1 = nn.Conv2d(1, 4, kernel_size=3)
        # conv1.weight.data.fill_(1)
        # conv1.bias.data.fill_(0)
        self.conv1 = nn.Sequential(
            conv1,
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        conv2 = nn.Conv2d(4, 1, kernel_size=3)
        # conv2.weight.data.fill_(1)
        # conv2.bias.data.fill_(0)
        self.conv2 = nn.Sequential(
            conv2,
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(25, 4),
        #     nn.ReLU(),
        # )

        self.sm = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1)
        x = self.sm(x)
        return x

    def test(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1)
        # x = self.sm(x)
        return x


class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
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
            torch.nn.Linear(50, 4),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, 
            -1
        )  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


batch_size = 64
learning_rate = 0.001
momentum = 0.5
EPOCH = 5

spec_val = 1
t1 = torch.tensor(
    [spec_val, 0, 0, 0],
    dtype=torch.float32,
)
t2 = torch.tensor(
    [0, spec_val, 0, 0],
    dtype=torch.float32,
)
t3 = torch.tensor(
    [0, 0, spec_val, 0],
    dtype=torch.float32,
)
t4 = torch.tensor(
    [0, 0, 0, spec_val],
    dtype=torch.float32,
)


# data = {t1: mat.a, t2: mat.b, t3: mat.c, t4: mat.d}
# data = {t1: mat.b, t2: mat.d, t3: mat.c, t4: mat.a}
# data = {t1: mat.b}#, t2: mat.d, t3: mat.c, t4: mat.a}
# data = {t2: mat.d,t2:mat.test_d}
# data = [[t2,mat.d],[t2,mat.test_d]]

data = [[t1, mat.bb], [t2, mat.dd], [t3, mat.cc], [t4, mat.aa]]
data = [[t1, mat.aa], [t2, mat.bb], [t3, mat.cc], [t4, mat.dd]]
data = [[t4, mat.aa], [t1, mat.bb], [t2, mat.cc], [t3, mat.dd]]
# data = [[t1,mat.bb]]

# for res, input in data.items():
#     print(res)

# model = Net()
model = Net1()

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # lr学习率，momentum冲量

# for i in range(0, mat.bas):
#     for val in data:
#         optimizer.zero_grad()
#         output = model(val[1][i])
#         loss = criterion(output, val[0])

#         print(loss)
#         loss.backward()
#         optimizer.step()


for nnx in range(0, 10000):
    i = random.randint(0, mat.bas-1)
    index = random.randint(0, 3)
    val = data[index]
    optimizer.zero_grad()
    output = model(val[1][i])
    loss = criterion(output, val[0])

    if nnx % 100 == 0:
        print(loss,i,index)

    loss.backward()
    optimizer.step()

# for param in model.parameters():
#     print(param)

input = mat.d
# print(model.test(input))
result = model(input)
print(result)
print(t1)
# result = model(mat.b)
# print(result)
# result = model(mat.c)
# print(result)
# result = model(mat.d)
# print(result)
