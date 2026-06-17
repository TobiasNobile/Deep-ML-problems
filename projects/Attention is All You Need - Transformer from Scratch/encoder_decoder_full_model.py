from feedforward_layernorm_dropout import *
from multihead_attention import *

def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    """
    Run multi-head self-attention on x and wrap with residual add-and-norm.
    """
    attention = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    result = apply_residual_add_and_norm(x, attention, gamma, beta)
    return result

def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    """
    Run the position-wise FFN on x and wrap it with residual add-and-norm.
    """
    ffn = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, ffn, gamma, beta)

def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    """
    Chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    """
    w_q, w_k, w_v = layer_params["w_q"], layer_params["w_k"], layer_params["w_v"]
    w_o, attn_gamma, attn_beta = layer_params["w_o"], layer_params["attn_gamma"], layer_params["attn_beta"]
    w1, b1, w2, b2 = layer_params["w1"], layer_params["b1"], layer_params["w2"], layer_params["b2"]
    ffn_gamma, ffn_beta = layer_params["ffn_gamma"], layer_params["ffn_beta"]

    self_attention = encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, attn_gamma, attn_beta, num_heads, src_mask)
    return encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, ffn_gamma, ffn_beta)

def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    """
    Sequentially apply each encoder layer to the running hidden state and return the final tensor.
    """
    state = x
    for i in range(len(encoder_layer_params_list)):
        layer_params = encoder_layer_params_list[i]
        state = assemble_encoder_layer(state, layer_params, num_heads, src_mask)
    return state