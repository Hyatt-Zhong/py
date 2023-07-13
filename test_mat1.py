import matrix as mat

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


batch_size = 64
learning_rate = 0.01
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

data = [[t1,mat.bb],[t2,mat.dd],[t3,mat.cc],[t4,mat.aa]]
data = [[t1,mat.bb]]

# for res, input in data.items():
#     print(res)

model = Net()

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate
)  # lr学习率，momentum冲量

for i in range(0, mat.bas):
    for val in data:
        optimizer.zero_grad()
        output = model(val[1][i])
        loss = criterion(output, val[0])

        print(loss)
        loss.backward()
        optimizer.step()


# for param in model.parameters():
#     print(param)

input = mat.d
print(model.test(input))
result = model(input)
print(result)
# result = model(mat.b)
# print(result)
# result = model(mat.c)
# print(result)
# result = model(mat.d)
# print(result)
