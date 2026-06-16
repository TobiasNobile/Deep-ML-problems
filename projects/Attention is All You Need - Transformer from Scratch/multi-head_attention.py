import torch

from masks_scaled_dot_product_attention import scaled_dot_product_attention

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

def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    """
    project x into separate query, key, and value tensors via three linear layers
    """
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)
    return q, k, v

def split_qkv_into_heads(q, k, v, num_heads):
    """
    split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    """
    q_h = split_last_dim_into_heads(q, num_heads)
    k_h = split_last_dim_into_heads(k, num_heads)
    v_h = split_last_dim_into_heads(v, num_heads)

    q_h = transpose_heads_before_sequence(q_h)
    k_h = transpose_heads_before_sequence(k_h)
    v_h = transpose_heads_before_sequence(v_h)
    
    return q_h, k_h, v_h

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    """
    run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    """
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

def merge_heads_and_project_output(context, w_o, b_o):
    """
    merge the head axis back into d_model and apply the output linear projection.
    """
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged, w_o, b_o)

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    """
    project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    Supports self-attention and cross-attention
    """
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    context, weights = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)
    return merge_heads_and_project_output(context, w_o, None)
