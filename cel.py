import numpy as np

import torch
import torch.nn as nn

def cross_entropy_error(y,t):
    delta=1e-7  #添加一个微小值可以防止负无限大(np.log(0))的发生。
    return -np.sum(t*np.log(y+delta))

def cross_entropy(y, y_hat):
    assert y.shape == y_hat.shape
    n = 1e-7
    res = -np.sum(np.nan_to_num(y * np.log(y_hat + n))) # 行求和 
    return res


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y1),np.array(t)))  #输出0.510825457099338（也比较小啦）

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y2),np.array(t)))  #2.302584092994546

tt=torch.tensor(t,dtype=torch.float32)
ty1=torch.tensor(y1,dtype=torch.float32)
ty2=torch.tensor(y2,dtype=torch.float32)

my_loss = cross_entropy(np.array(y2),np.array(t))
print(my_loss)

CEL = nn.CrossEntropyLoss()

print(CEL(ty1, tt))
print(CEL(ty2, tt))
