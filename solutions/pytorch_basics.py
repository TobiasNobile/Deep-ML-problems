import torch

def to_float_tensor(values):
    """
    Return a torch.float32 tensor built from `values`
    """
    return torch.tensor(values, dtype=torch.float32)

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    """
    Flatten x to 1-D, then rearrange into new_shape
    """
    return torch.reshape(torch.flatten(x), new_shape)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    """
    Swap the last two dimensions of x
    """
    return torch.transpose(x, -1, -2)