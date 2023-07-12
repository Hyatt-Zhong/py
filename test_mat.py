import matrix as mat

import torch.nn as nn
import torch

# pip3 install torch torchvision torchaudio
# 2.0.1+cpu


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conv1 = torch.nn.Conv2d(1, 1, kernel_size=5)
        conv1.weight.data.fill_(1)
        conv1.bias.data.fill_(0)
        self.conv1 = torch.nn.Sequential(
            conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        conv2 = torch.nn.Conv2d(1, 1, kernel_size=1)
        conv2.weight.data.fill_(1)
        conv2.bias.data.fill_(0)
        self.conv2 = torch.nn.Sequential(
            conv2,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        spec_fc = nn.Linear(25, 4)
        spec_fc.weight.data.fill_(1)
        spec_fc.bias.data.fill_(0)
        self.fc = nn.Sequential(
            spec_fc,
            nn.ReLU(),
        )

        self.sm = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = x.view(-1)
        x = self.fc(x)
        x = self.sm(x)
        return x

    def test(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
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
data = {t1: mat.a, t2: mat.b, t3: mat.c, t4: mat.d}
# data = {t2: mat.b}
# data = {t3: mat.c}
# data = {t4: mat.d}

# for res, input in data.items():
#     print(res)

model = Net()
# print("############################")
# for parameters in model.parameters():
#     print(parameters)

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum
)  # lr学习率，momentum冲量

for i in range(0, 10):
    for res, input in data.items():
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, res)

        print(loss)
        loss.backward()
        optimizer.step()


# print(criterion(t4, t4))
# print("############################")
for parameters in model.parameters():
    print(parameters)

input = mat.test_val
result = model(input)
print(result.size())
print(result)
print(model.test(input))
# result = model(mat.b)
# print(result)
# result = model(mat.c)
# print(result)
# result = model(mat.d)
# print(result)

# tn = torch.tensor(
#     [0.0000, 3188.0000, 3188.0000, 3188.7444],
#     dtype=torch.float32,
# )

# print(criterion(tn, t4))
# print(criterion(t4, t4))
