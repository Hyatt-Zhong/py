import matrix as mat

import torch.nn as nn
import torch
# pip3 install torch torchvision torchaudio
# 2.0.1+cpu


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(1, 1, kernel_size=4),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(25, 4),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1)
        x = self.fc(x)
        return x


# mynet = Net()
# y = mynet(mat.a)

# print(y)

batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 5

spec_val = 5
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
data = {t1: mat.a}  # , t2: mat.b, t3: mat.c, t4: mat.d}

# for res, input in data.items():
#     print(res)

model = Net()

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum
)  # lr学习率，momentum冲量

for i in range(0, 10):
    for res, input in data.items():
        # optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, res)

        print(loss)
        loss.backward()
        optimizer.step()


# print(criterion(t4, t4))

result = model(mat.a)
print(result)
# result = model(mat.b)
# print(result)
# result = model(mat.c)
# print(result)
# result = model(mat.d)
# print(result)
