import torch
import numpy as np
import torch.nn as nn

# sz_pic_input = 1000 #图片输入维度
sz_pic = 128 #图片处理维度
sz_con = 32 #概念维度
# heads = 8 #注意多少个头

sz_context = sz_con #语境空间维度

class Context(nn.Module): #语境
    def __init__(self):
        super(Context, self).__init__()

    def forward(self, x):
        return x
    
    
class Control(nn.Module):#
    def __init__(self):
        super(Control, self).__init__()
        
    def forward(self, context, x):
        return x
    

class Attention(nn.Module):#
    def __init__(self):
        super(Attention, self).__init__()
        
    def forward(self, context, x):
        return x
        

class PicNet(nn.Module):
    def __init__(self):
        super(PicNet, self).__init__()
        self.att = Attention()

    def forward(self, context, x):
        return x  
    

class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()

    def forward(self, context, x):
        return x  


# class Concept(nn.Module): #概念
#     def __init__(self):
#         super(Concept, self).__init__()

#     def forward(self, x):
#         return x

# class Cut(nn.Module):#分割 比如分割前景背景 图块分割
#     def __init__(self, context):
#         super(Cut, self).__init__()

#     def forward(self, x):
#         return x

# class Control(nn.module):
#     def __init__(self):
#         super(Control, self).__init__()

#     def forward(self, x):
#         return x
