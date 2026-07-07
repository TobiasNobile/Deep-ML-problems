import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        B, T, d_model = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = torch.reshape(Q, (B, T, self.num_heads, self.d_head))
        K = torch.reshape(K, (B, T, self.num_heads, self.d_head))
        V = torch.reshape(V, (B, T, self.num_heads, self.d_head))

        Q = torch.permute(Q, (0, 2, 1, 3))
        K = torch.permute(K, (0, 2, 1, 3))
        V = torch.permute(V, (0, 2, 1, 3))


        dot = Q @ torch.transpose(K, -2, -1) /self.d_head**0.5

        if mask is not None:
            dot += mask
        softed = F.softmax(dot, dim=-1)
        softed @= V
        softed = torch.permute(softed, (0, 2, 1, 3))
        softed = torch.reshape(softed, (B, T, d_model))

        return self.out_proj(softed)
