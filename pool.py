import torch.nn as nn
import torch

im = torch.tensor(
    [
        [
            [
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 5, 5],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 5, 5],
                [0, 1, 0, 0, 0, 0],
            ]
        ]
    ],
    dtype=torch.float32,
)


class net(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(net, self).__init__()

        self.c = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.c(x)
        return out


mynet = net()
y = mynet(im)

print(y)
