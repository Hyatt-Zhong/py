import torch.nn as nn
import torch

# # 保存模型及相关信息
# model_info = {
#     'input_size': input_size,
#     'hidden_size': hidden_size,
#     'output_size': output_size,
#     'state_dict': dynamic_net.state_dict()
# }
# torch.save(model_info, 'dynamic_model_info.pth')

# # 加载模型及相关信息
# loaded_model_info = torch.load('dynamic_model_info.pth')
# loaded_model = DynamicNet(loaded_model_info['input_size'], loaded_model_info['hidden_size'], loaded_model_info['output_size'])
# loaded_model.load_state_dict(loaded_model_info['state_dict'])

# 创建一个张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

y =torch.tensor([[1, 1, 0], [0, 0, 1]])

# 将x与随机张量相乘
result = x * y

print(result)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.arr_count=0
        self.knowlege=[]


    def new_array(self):
        self.knowlege.append(torch.ones(1,100,10,10))
        self.arr_count+=1
        return self.knowlege[self.arr_count-1]
    
    def save(self):
        net_info={
            'arr_count':self.arr_count,
            'knowlege':self.knowlege
        }
        torch.save(net_info,"net_info.pth")
    
    def load(self):
        net_info=torch.load("net_info.pth")
        self.arr_count=net_info['arr_count']
        self.knowlege=net_info['knowlege']


    def forward(self, x):
        x = self.conv1(x)
        # x = x.view(-1)
        # x = self.fc1(x)
        return x
    
net = Net()
# net.new_array()
net.load()
print(net.knowlege[0][0,0])
# net.save()
