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

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    """
    Run masked multi-head self-attention on y and wrap with residual add-and-norm.
    """
    attention = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, attention, gamma, beta)

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    """
    Run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    """
    if src_mask is not None:
        B, L = src_mask.shape
        src_mask = torch.reshape(src_mask, (B, 1, 1, L))
    attention = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(y, attention, gamma, beta)

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    """
    Run the position-wise FFN on y and wrap it with residual add-and-norm
    """
    ffn = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, ffn, gamma, beta)

def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """
    Run a full decoder layer: masked self-attention, cross-attention, then FFN.
    """

    w_q_self =  layer_params["w_q_self"]
    w_k_self =  layer_params["w_k_self"]
    w_v_self =  layer_params["w_v_self"]
    w_o_self =  layer_params["w_o_self"]
    self_gamma =  layer_params["self_gamma"]
    self_beta =  layer_params["self_beta"]

    w_q_cross =  layer_params["w_q_cross"]
    w_k_cross =  layer_params["w_k_cross"]
    w_v_cross =  layer_params["w_v_cross"]
    w_o_cross =  layer_params["w_o_cross"]
    cross_gamma =  layer_params["cross_gamma"]
    cross_beta =  layer_params["cross_beta"]

    w1, b1 =  layer_params["w1"], layer_params["b1"]
    w2, b2 =  layer_params["w2"], layer_params["b2"]
    ffn_gamma, ffn_beta =  layer_params["ffn_gamma"], layer_params["ffn_beta"]

    self_attention = decoder_layer_masked_self_attention_sublayer(y, w_q_self, w_k_self, w_v_self, w_o_self, self_gamma, self_beta, num_heads, tgt_mask)
    cross_attention = decoder_layer_cross_attention_sublayer(self_attention, encoder_output, w_q_cross, w_k_cross, w_v_cross, w_o_cross, cross_gamma, cross_beta, num_heads, src_mask)
    ffn = decoder_layer_feed_forward_sublayer(cross_attention, w1, b1, w2, b2, ffn_gamma, ffn_beta)
    return ffn