import math

import torch

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """
    Return a dict of leaf tensors with requires_grad=True for one encoder layer.
    Allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    """
    layer_parameters = {}
    layer_parameters["w_q"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_k"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_v"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_o"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w1"] = (torch.randn(d_model, d_ff)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w2"] = (torch.randn(d_ff, d_model)*0.5).requires_grad_(requires_grad=True)

    layer_parameters["attn_gamma"] = torch.ones(d_model, requires_grad=True)
    layer_parameters["attn_beta"] = torch.zeros(d_model, requires_grad=True)
    layer_parameters["ffn_gamma"] = torch.ones(d_model, requires_grad=True)
    layer_parameters["ffn_beta"] = torch.zeros(d_model, requires_grad=True)

    layer_parameters["b1"] = torch.zeros(d_ff, requires_grad=True)
    layer_parameters["b2"] = torch.zeros(d_model, requires_grad=True)

    return layer_parameters





