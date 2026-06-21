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

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    """
    Return a dict of requires_grad tensors for one decoder layer
    """
    layer_parameters = {}
    layer_parameters["w_q_self"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_k_self"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_v_self"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_o_self"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)

    layer_parameters["w_q_cross"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_k_cross"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_v_cross"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w_o_cross"] = (torch.randn(d_model, d_model)*0.5).requires_grad_(requires_grad=True)

    layer_parameters["w1"] = (torch.randn(d_model, d_ff)*0.5).requires_grad_(requires_grad=True)
    layer_parameters["w2"] = (torch.randn(d_ff, d_model)*0.5).requires_grad_(requires_grad=True)

    layer_parameters["b1"] = torch.zeros(d_ff, requires_grad=True)
    layer_parameters["b2"] = torch.zeros(d_model, requires_grad=True)

    layer_parameters["self_gamma"] = torch.ones(d_model, requires_grad=True)
    layer_parameters["self_beta"] = torch.zeros(d_model, requires_grad=True)

    layer_parameters["cross_gamma"] = torch.ones(d_model, requires_grad=True)
    layer_parameters["cross_beta"] = torch.zeros(d_model, requires_grad=True)

    layer_parameters["ffn_gamma"] = torch.ones(d_model, requires_grad=True)
    layer_parameters["ffn_beta"] = torch.zeros(d_model, requires_grad=True)

    return layer_parameters

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """
    Allocate src/tgt embeddings and output projection (optionally tied).
    """
    parameters = {}
    if tie_weights:
        parameters["tgt_embedding"] = torch.randn(vocab_size, d_model, requires_grad=True)
        parameters["output_projection"] = parameters["tgt_embedding"]
    else:
        parameters["tgt_embedding"] = torch.randn(vocab_size, d_model, requires_grad=True)
        parameters["output_projection"] = torch.randn(vocab_size, d_model, requires_grad=True)
    parameters["src_embedding"] = torch.randn(vocab_size, d_model, requires_grad=True)
    
    return parameters

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    """
    Walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    """
    parameters_list = []
    for d in (encoder_layer_params, decoder_layer_params):
        for layer in d:
            parameters_list.extend(layer.values())
    parameters_list.extend(embedding_params.values())

    param_dd = {id(v):v for v in parameters_list}
    return list(param_dd.values())



