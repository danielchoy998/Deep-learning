import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        # 線性投影（W_q, W_k, W_v）-> embed_size = size of the weight matrix
        self.query = nn.Linear(embed_size, embed_size) 
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.scale = embed_size ** 0.5  # √d_k for scaling

    def forward(self, x):
        # x: [batch_size, seq_len, embed_size]
        Q = self.query(x)  # [B, T, D]
        K = self.key(x)    # [B, T, D]
        V = self.value(x)  # [B, T, D]

        # 1. 計算 Attention score: Q x K^T
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # [B, T, T]

        # 2. softmax
        attention = torch.softmax(scores, dim=-1)  # [B, T, T]

        # 3. weighted sum: Attention x V
        out = torch.matmul(attention, V)  # [B, T, D]
        return out

embed_size = 64
seq_len = 10
batch_size = 2

x = torch.randn(batch_size, seq_len, embed_size)
attention = SelfAttention(embed_size)
out = attention(x)

print("Output shape:", out.shape) 