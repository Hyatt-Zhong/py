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

sm = nn.Softmax(dim=0)
print(sm(alpha))
CEL = nn.CrossEntropyLoss()

print(CEL(b, a))
print(CEL(a, b))
print(CEL(zeros,ones))
print(CEL(ones,zeros))

