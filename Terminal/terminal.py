import torch
import numpy as np
import torch.nn as nn

con_size = 100
pic_size = 100

class PicNet(nn.Module):
    def __init__(self):
        super(PicNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 30, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(2430, 320),
            nn.Linear(320, con_size),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x)  
        x = x.view(batch_size, -1)  
        x = self.fc(x)
        return x 
    
# input = torch.randn(1, pic_size, pic_size)
# mod = PicNet()
# output = mod(input)
# print(output)
