import math

import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    B, L = token_ids.shape

    mask = token_ids != pad_id 

    return torch.reshape(mask, (B, 1, 1, L))

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    mask = torch.zeros((seq_len, seq_len), dtype = torch.bool)
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                mask[i, j] = True

    return torch.reshape(torch.tril(mask), (1, 1, seq_len, seq_len))

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    """
    Combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    """
    return padding_mask & causal_mask

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    Kt = torch.transpose(key, -1, -2)
    return torch.matmul(query, Kt)

def scale_attention_scores(scores, d_k):
    """divide raw attention scores by sqrt(d_k) to stabilize softmax inputs"""
    return scores / math.sqrt(d_k)

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return scores.masked_fill(~mask, float('-inf'))

def softmax_attention_weights(masked_scores):
    """
    softmax over the last axis, zeroing rows that are entirely -inf
    """
    
    fully_masked = torch.all(masked_scores == float('-inf'), dim=-1, keepdim=True)
    zeros = torch.zeros(masked_scores.shape)
    softmax_result = torch.softmax(masked_scores, dim=-1)
    softmax_result = torch.where(fully_masked, zeros, softmax_result)
    return softmax_result

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    return attention_weights @ value

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    Kt = torch.transpose(key, -1, -2)

    scores = torch.matmul(query, Kt) / math.sqrt(key.shape[-1])
    masked_scores = scores.masked_fill(~mask, float('-inf')) if mask is not None else scores

    fully_masked = torch.all(masked_scores == float('-inf'), dim=-1, keepdim=True)
    zeros = torch.zeros(masked_scores.shape)
    softmax_result = torch.softmax(masked_scores, dim=-1)
    softmax_result = torch.where(fully_masked, zeros, softmax_result)
    return softmax_result @ value, softmax_result