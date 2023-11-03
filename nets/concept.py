import sys 
sys.path.append("..") 
import torch.nn as nn
from py import matrix as mat


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conv_ = nn.Conv2d(1, 1, kernel_size=4)
        conv_.weight.data.fill_(1)
        conv_.bias.data.fill_(0)
        self.conv1 = nn.Sequential(
            conv_,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(25, 5)

    def forward(self, x):
        x = self.conv1(x)
        # x = x.view(-1)
        # x = self.fc1(x)
        return x


