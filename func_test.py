import torch
import torch.nn as nn
import copy

a = torch.tensor(
    [1],
    dtype=torch.float32,
)
b = torch.tensor(
    [0.5],
    dtype=torch.float32,
)

alpha = torch.tensor(
    [1, 1, 1, 5],
    dtype=torch.float32,
)
beta = torch.tensor(
    [1, 2, 3, 4],
    dtype=torch.float32,
)

zeros5 = torch.zeros(1, 1, 5, 5)
gamma = zeros5
gamma[0][0][:, 2] = 1

delta = copy.deepcopy(zeros5)
delta[0][0][:, 2] = -1


# print(gamma)

zeros = torch.zeros(4)
ones = torch.ones(4)


def functest1(name, fun, x=torch.randn(1)):
    print(name + ":")
    y = fun(x)
    print("input:", x, "\noutput:", y, "\n")
    print("------------")


def functest2(name, fun, m, n):
    print(name + ":")
    y = fun(m, n)
    print(y)
    print("------------")


def conv_relu(x):
    conv1 = nn.Conv2d(1, 1, kernel_size=2)
    conv1.weight.data.fill_(1)
    conv1.bias.data.fill_(0)
    conv = nn.Sequential(
        conv1,
        nn.ReLU()
    )
    return conv(x)


functest1("sigmoid", nn.Sigmoid())
functest1("relu", nn.ReLU(), torch.rand(4))
functest1("relu", nn.ReLU(), delta)
functest1("softmax", nn.Softmax(dim=0), alpha)
functest1("conv_relu", conv_relu, gamma)
functest1("conv_relu", conv_relu, delta)


functest2("CEL", nn.CrossEntropyLoss(), b, a)
functest2("CEL", nn.CrossEntropyLoss(), a, b)
functest2("CEL", nn.CrossEntropyLoss(), zeros, ones)
functest2("CEL", nn.CrossEntropyLoss(), ones, zeros)
