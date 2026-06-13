import torch

def split_last_dim_into_heads(tensor, num_heads):
    """
    reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    """
    B, L, d_model = tensor.shape
    return torch.reshape(tensor, (B, L, num_heads, d_model // num_heads))

def transpose_heads_before_sequence(split_tensor):
    """
    rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    """
    return torch.permute(split_tensor, (0, 2, 1, 3))