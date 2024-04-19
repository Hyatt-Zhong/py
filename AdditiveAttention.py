import torch
import torch.nn.functional as F

class AdditiveAttention(torch.nn.Module):
    def __init__(self, input_size):
        super(AdditiveAttention, self).__init__()
        self.Wq = torch.nn.Linear(input_size, input_size)
        self.Wk = torch.nn.Linear(input_size, input_size)
        self.Wv = torch.nn.Linear(input_size, input_size)

    def forward(self, input_seq):
        query = self.Wq(input_seq)
        key = self.Wk(input_seq)
        value = self.Wv(input_seq)

        score = torch.matmul(query, key.transpose(1, 2))
        weights = F.softmax(score, dim=-1)

        output = torch.matmul(weights, value)
        return output

# 示例
input_size = 4
seq_length = 5
input_seq = torch.rand(1, seq_length, input_size)  # 输入序列

attention = AdditiveAttention(input_size)
output = attention(input_seq)

print("Input sequence:")
print(input_seq)
print("\nOutput sequence after attention:")
print(output)