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

def merge_heads_back_to_model_dim(multi_head_tensor):
    """
    merge the head axis back into the feature axis to reconstruct d_model
    """
    merged_heads = torch.transpose(multi_head_tensor, 1, 2)
    B, L, H, d_k = merged_heads.shape
    return torch.reshape(merged_heads, (B, L, H*d_k))

def apply_linear_projection(x, weight, bias):
    """
    return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    """
    if bias is not None:
        return x @ weight.T + bias
    return x @ weight.T
