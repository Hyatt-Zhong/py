import torch.nn as nn
import torch
import matrix as mat
from drawmatrx import drawmtx,show

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input):
        output = self.model(input)
        return output

mynet=Net()


im=mat.d
print(im)
drawmtx(im[0][0])

output = mynet(im)

print(output)
drawmtx(output[0][0].detach())

print(mynet.parameters)

