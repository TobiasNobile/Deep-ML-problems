import torch

def to_float_tensor(values):
    """
    Return a torch.float32 tensor built from `values`
    """
    return torch.tensor(values, dtype=torch.float32)