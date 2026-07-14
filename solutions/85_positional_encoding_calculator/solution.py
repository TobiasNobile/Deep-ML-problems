import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    if position == 0 or d_model <= 0:
        return -1
    pos_encoding = torch.zeros((position, d_model), dtype=torch.float16)
    for pos in range(len(pos_encoding)):
        for i in range(pos_encoding.shape[1] // 2):
            angle = torch.tensor(pos / 10000**(2*i/d_model))
            pos_encoding[pos, 2*i]   = torch.sin(angle)
            pos_encoding[pos, 2*i+1] = torch.cos(angle)
    return pos_encoding
