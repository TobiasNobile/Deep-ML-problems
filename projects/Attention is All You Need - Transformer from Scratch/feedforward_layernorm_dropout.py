import torch

def apply_ffn_first_linear_and_relu(x, w1, b1):
    """
    project x by w1, add b1, then apply a ReLU activation.
    """
    return torch.relu(x @ w1 + b1)

def apply_ffn_second_linear(hidden, w2, b2):
    """
    project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    """
    return hidden @ w2 + b2

def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    """
    Compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    """
    L1 = apply_ffn_first_linear_and_relu(x, w1, b1)
    L2 =  apply_ffn_second_linear(L1, w2, b2)
    return L2

def compute_layer_norm_mean_and_variance(x):
    """
    Return (mean, variance) reduced over the last dim with shape (..., 1)
    """
    return torch.mean(x, keepdim = True, dim = -1), torch.var(x, dim=-1, keepdim = True, correction = 0)

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    """
    Standardize x along the last axis then apply gamma and beta affine transform
    """
    mean, var = compute_layer_norm_mean_and_variance(x)
    return gamma*(x - mean)/(torch.sqrt(var + eps)) + beta

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    """
    combine the residual with the sublayer output and layer-normalize the result.
    """
    return  normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps=1e-5)

def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    """
    Multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    """
    return torch.mul(x, keep_mask.to(x.dtype))  / keep_prob