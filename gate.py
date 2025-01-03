# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GatingNetwork(nn.Module):
#     def __init__(self, input_size):
#         super(GatingNetwork, self).__init__()
#         self.fc = nn.Linear(input_size, 2)  # 输出两个权重

#     def forward(self, x):
#         gate_weights = F.softmax(self.fc(x), dim=1)  # 使用softmax生成权重
#         print(x)
#         print(self.fc(x))
#         print(gate_weights)

#         return gate_weights

# class ExpertNetwork1(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(ExpertNetwork1, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# class ExpertNetwork2(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(ExpertNetwork2, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# class MixtureOfExperts(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MixtureOfExperts, self).__init__()
#         self.gating_network = GatingNetwork(input_size)
#         self.expert1 = ExpertNetwork1(input_size, hidden_size, output_size)
#         self.expert2 = ExpertNetwork2(input_size, hidden_size, output_size)

#     def forward(self, x):
#         gate_weights = self.gating_network(x)
#         expert1_output = self.expert1(x)
#         expert2_output = self.expert2(x)

#         # 门控网络输出的权重
#         output = gate_weights[:, 0].unsqueeze(1) * expert1_output + gate_weights[:, 1].unsqueeze(1) * expert2_output
#         return output

# # 示例输入
# input_size = 10
# hidden_size = 20
# output_size = 1
# batch_size = 1

# model = MixtureOfExperts(input_size, hidden_size, output_size)
# input_data = torch.randn(batch_size, input_size)
# output = model(input_data)
# print(output)
