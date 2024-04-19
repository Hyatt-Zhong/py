import torch
import torch.nn as nn
import random

def print_x(p):
    # print(p)
    pass

spec_val=10
spec_size=29
spec_a=3
spec_b=spec_size-spec_a

alpha=torch.ones(1,1,5,5)
alpha[0][0][1:spec_a,1:spec_a]=1

a=torch.ones(1,1,spec_size,spec_size)
a[0][0][:spec_a,:spec_a]=spec_val
print_x(a)

b=torch.ones(1,1,spec_size,spec_size)
b[0][0][:spec_a,spec_b:]=spec_val
print_x(b)

c=torch.ones(1,1,spec_size,spec_size)
c[0][0][spec_b:,:spec_a]=spec_val
print_x(c)

d=torch.ones(1,1,spec_size,spec_size)
d[0][0][spec_b:,spec_b:]=spec_val
print_x(d)

test_val=d
d[0][0][10:,10:]=spec_val


_ii=15
e=torch.ones(1,1,spec_size,spec_size)
e[0][0][:spec_a,:spec_a]=_ii
print_x(e)

f=torch.ones(1,1,spec_size,spec_size)
f[0][0][:spec_a,spec_b:]=_ii
print_x(f)

g=torch.ones(1,1,spec_size,spec_size)
g[0][0][spec_b:,:spec_a]=_ii
print_x(g)

h=torch.ones(1,1,spec_size,spec_size)
h[0][0][spec_b:,spec_b:]=_ii
print_x(h)

test_d=d
d[0][0][11:,11:]=spec_val+1


bas=40
aa=torch.zeros(bas,1,1,spec_size,spec_size)
bb=torch.zeros(bas,1,1,spec_size,spec_size)
cc=torch.zeros(bas,1,1,spec_size,spec_size)
dd=torch.zeros(bas,1,1,spec_size,spec_size)
# aa=torch.ones(bas,1,1,spec_size,spec_size)
# bb=torch.ones(bas,1,1,spec_size,spec_size)
# cc=torch.ones(bas,1,1,spec_size,spec_size)
# dd=torch.ones(bas,1,1,spec_size,spec_size)
for i in range(bas):
    aa[i][0][0][:spec_a,:spec_a]=random.randint(7,13)
    bb[i][0][0][:spec_a,spec_b:]=random.randint(7,13)
    cc[i][0][0][spec_b:,:spec_a]=random.randint(7,13)
    dd[i][0][0][spec_b:,spec_b:]=random.randint(7,13)
    # aa[i][0][0][:spec_a,:spec_a]=spec_val
    # bb[i][0][0][:spec_a,spec_b:]=spec_val
    # cc[i][0][0][spec_b:,:spec_a]=spec_val
    # dd[i][0][0][spec_b:,spec_b:]=spec_val
    for i in range(4):
        xx=random.randint(0, spec_size-1)
        yy=random.randint(0, spec_size-1)
        aa[i][0][0][xx][yy]=random.randint(7,13)
        xx=random.randint(0, spec_size-1)
        yy=random.randint(0, spec_size-1)
        bb[i][0][0][xx][yy]=random.randint(7,13)
        xx=random.randint(0, spec_size-1)
        yy=random.randint(0, spec_size-1)
        cc[i][0][0][xx][yy]=random.randint(7,13)
        xx=random.randint(0, spec_size-1)
        yy=random.randint(0, spec_size-1)
        dd[i][0][0][xx][yy]=random.randint(7,13)


# print(aa[i][0][0])
# print(bb[i][0][0])
# print(cc[i][0][0])
# print(dd[i][0][0])
