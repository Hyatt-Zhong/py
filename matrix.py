import torch
import torch.nn as nn

a=torch.zeros(1,1,5,5)
print(a)
a[0][0][1:3,1:3]=1
print(a)