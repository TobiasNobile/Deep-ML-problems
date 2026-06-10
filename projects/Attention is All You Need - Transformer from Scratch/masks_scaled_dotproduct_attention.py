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