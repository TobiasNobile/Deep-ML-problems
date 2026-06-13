import torch

def split_last_dim_into_heads(tensor, num_heads):
    """
    reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    """
    B, L, d_model = tensor.shape
    return torch.reshape(tensor, (B, L, num_heads, d_model // num_heads))
