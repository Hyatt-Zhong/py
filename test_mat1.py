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
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = x.view(-1)
        return x


# mynet = Net()
# y = mynet(mat.a)

# print(y)

batch_size = 64
learning_rate = 0.05
momentum = 0.5
EPOCH = 5

t1 = torch.tensor(
    [1,0,0,0],
    dtype=torch.float32,
)
t2 = torch.tensor(
    [0,1,0,0],
    dtype=torch.float32,
)
t3 = torch.tensor(
    [0,0,1,0],
    dtype=torch.float32,
)
t4 = torch.tensor(
    [0,0,0,1],
    dtype=torch.float32,
)
data = {t1: mat.e, t2: mat.f, t3: mat.g, t4: mat.h}

# for res, input in data.items():
#     print(res)

model = Net()

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum
)  # lr学习率，momentum冲量

# for i in range(0, 100):
#     for res, input in data.items():
#         optimizer.zero_grad()
#         output = model(input)
#         loss = criterion(output, res)

#         loss.backward()
#         optimizer.step()


t44 = torch.tensor(
    [[1]],
    dtype=torch.float32,
)
t45 = torch.tensor(
    [0],
    dtype=torch.long,
)
output = model(mat.h)
print(output,t4)
loss = criterion(t44, t45)

nxx = model(mat.h)
print(nxx)
