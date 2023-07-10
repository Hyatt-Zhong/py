import torch.nn as nn
import matrix as mat


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(25, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1)
        x = self.fc1(x)
        return x


model = Net()
nxx = model(mat.d)
print(nxx)