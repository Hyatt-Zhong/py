import torch
import torch.nn as nn

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

zeros=torch.zeros(4)
ones=torch.ones(4)

def functest1(name, fun, x=torch.randn(1)):
    print(name+":")
    y=fun(x)
    print(x,y)
    print("------------")

def functest2(name, fun, m,n):
    print(name+":")
    y=fun(m,n)
    print(y)
    print("------------")

functest1("sigmoid",nn.Sigmoid()) 
functest1("relu",nn.ReLU(),torch.rand(4)) 
functest1("softmax",nn.Softmax(dim=0),alpha) 

functest2("CEL",nn.CrossEntropyLoss(),b,a)
functest2("CEL",nn.CrossEntropyLoss(),a,b)
functest2("CEL",nn.CrossEntropyLoss(),zeros,ones)
functest2("CEL",nn.CrossEntropyLoss(),ones,zeros)