
import torch.nn as nn
import numpy as np
import torch
import cv2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conv_ = nn.Conv2d(1, 1, kernel_size=1)
        conv_.weight.data.fill_(1)
        # conv_.weight.data=torch.tensor([[[[0., -1., 0.],
        #                                   [-1., 4.,-1.],
        #                                   [0., -1., 0.]]]])
        conv_.bias.data.fill_(0)
        self.conv1 = nn.Sequential(
            # conv_,
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.MaxPool2d(kernel_size=2),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(25, 5)

    def forward(self, x):
        x = self.conv1(x)
        # x = x.view(-1)
        # x = self.fc1(x)
        return x

def read_gray_image(pic):
    image = cv2.imread(pic)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def show_tensor(t):
    print(t.shape)
    p=t.to(torch.uint8)
    numpy_array = p.detach().numpy()
    cv2.imshow('Result', numpy_array)
    cv2.waitKey(0)

img = read_gray_image('desktop.jpg')
tensor=torch.from_numpy(img)
show_tensor(tensor)

input=tensor.reshape(1,1,img.shape[0],img.shape[1])
input=input.to(torch.float)


model = Net()
out = model(input)
ret=out.reshape(out.shape[2],out.shape[3])
show_tensor(ret)

torch.save(model.state_dict())

cv2.destroyAllWindows()
