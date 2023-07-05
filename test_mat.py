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
        )

    def forward(self, x):
        x = self.conv1(x)  
        return x 


# mynet = Net()
# y = mynet(mat.a)

# print(y)

batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 5

data ={1:mat.a,2:mat.b,3:mat.c,4:mat.d}

for res,input in data.items():
    print(res)

model = Net()
 
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量

