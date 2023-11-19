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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=5),
            # torch.nn.ReLU(),
        )

    def forward(self, input):
        output = self.model(input)
        return output

mynet=Net()
y = mynet(im)

print(y)
