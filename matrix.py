import torch
import torch.nn as nn

def print_x(p):
    # print(p)
    pass


alpha=torch.zeros(1,1,5,5)
alpha[0][0][1:3,1:3]=1

a=torch.zeros(1,1,15,15)
a[0][0][:3,:3]=3
print_x(a)

b=torch.zeros(1,1,15,15)
b[0][0][:3,12:]=3
print_x(b)

c=torch.zeros(1,1,15,15)
c[0][0][12:,:3]=3
print_x(c)

d=torch.zeros(1,1,15,15)
d[0][0][12:,12:]=3
print_x(d)


_ii=15
e=torch.ones(1,1,15,15)
e[0][0][:3,:3]=_ii
print_x(e)

f=torch.ones(1,1,15,15)
f[0][0][:3,12:]=_ii
print_x(f)

g=torch.ones(1,1,15,15)
g[0][0][12:,:3]=_ii
print_x(g)

h=torch.ones(1,1,15,15)
h[0][0][12:,12:]=_ii
print_x(h)