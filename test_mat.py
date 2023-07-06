import matrix as mat

import torch.nn as nn
import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.MaxPool2d(kernel_size=3),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1)
        return x


# mynet = Net()
# y = mynet(mat.a)

# print(y)

batch_size = 64
learning_rate = 0.05
momentum = 0.5
EPOCH = 5

t1 = torch.tensor(
    [1],
    dtype=torch.float32,
)
t2 = torch.tensor(
    [2],
    dtype=torch.float32,
)
t3 = torch.tensor(
    [3],
    dtype=torch.float32,
)
t4 = torch.tensor(
    [4],
    dtype=torch.float32,
)
data = {t1: mat.a, t2: mat.b, t3: mat.c, t4: mat.d}

for res, input in data.items():
    print(res)

model = Net()

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum
)  # lr学习率，momentum冲量

for i in range(0, 1000):
    for res, input in data.items():
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, res)

        loss.backward()
        optimizer.step()


nxx = model(mat.d)
print(nxx)
