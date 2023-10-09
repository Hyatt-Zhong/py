
import torch
import torch.nn as nn

alpha=torch.ones(3,3,dtype=torch.int8)
print(alpha)

class NMove(nn.Module):
    def __init__(self):
        super(NMove, self).__init__()

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


