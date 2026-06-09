import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    return embeddings*math.sqrt(d_model)

def compute_positional_div_term(d_model):
    """
    return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    """
    return torch.FloatTensor([10000**(-2*i/d_model) for i in range(0, d_model//2)])

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    return torch.reshape(torch.arange(max_len, dtype=torch.float), (max_len, 1))

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    pe[:,::2] = torch.sin(position * div_term)
    return pe

def fill_odd_indices_with_cos(pe, position, div_term):
    """Fill odd feature indices of pe with cos(position * div_term)."""
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe