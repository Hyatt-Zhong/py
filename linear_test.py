import torch
import numpy as np
import torch.nn as nn

x1 = torch.rand([10,1])  + 1
print(x1)
y1 = x1 * 3 +5
print(y1)
x2 = torch.rand(10) * 99 + 1
y2 = x2 * 2 +7
# x1=x1.unsqueeze(1)
# y1=y1.unsqueeze(1)
# x2=x2.unsqueeze(1)
# y2=y2.unsqueeze(1)

class ln(nn.Module): #语境
    def __init__(self):
        super(ln, self).__init__()
        self.fc = nn.Linear(1,1)
        self.fc = nn.Sequential(nn.Linear(1,2),nn.Linear(2,1))

    def forward(self, x):
        return self.fc(x)
    

model = ln()

y = model(x1[0])

# print(y)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10000
for epoch in range(num_epochs):
    inputs = x1
    labels = y1


    outputs = model(inputs)
    # print(inputs)
    loss = criterion(outputs, labels)#这里和单权重的线性回归不一样
    optimizer.zero_grad()

    # print(loss)
    loss.backward()
    optimizer.step()

print(model(torch.tensor([5.0])))
for param in model.parameters():
    print(param)