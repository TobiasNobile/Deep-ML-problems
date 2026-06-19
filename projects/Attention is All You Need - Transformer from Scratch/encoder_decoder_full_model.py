import torch

from embeddings_positional_encoding import *
from masks_scaled_dot_product_attention import *
from multihead_attention import *
from feedforward_layernorm_dropout import *


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

def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    """
    Sequentially apply each decoder layer to the running target hidden state.
    """
    if not decoder_layer_params_list:
        return y
    decoder = assemble_decoder_layer(y, encoder_output, decoder_layer_params_list[0], num_heads, src_mask, tgt_mask)
    for layer_params in decoder_layer_params_list[1:]:
        decoder = assemble_decoder_layer(decoder, encoder_output, layer_params, num_heads, src_mask, tgt_mask)
    return decoder

def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    """
    Project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    """
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    return token_embedding_weight.T

def apply_log_softmax_over_vocab(logits):
    """
    Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    """
    return torch.nn.functional.log_softmax(logits, dim=-1)

def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    """
    Embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    """
    
    src_mask =  build_padding_mask(src_ids, pad_id)
    tgt_mask =  build_padding_mask(tgt_ids, pad_id)

    token_embedding = model_params['token_embedding']
    src_lookup = token_embedding[src_ids]
    tgt_lookup = token_embedding[tgt_ids]

    scaled_embed = scale_embeddings_by_sqrt_d_model(src_lookup, src_lookup.shape[-1])
    pe = build_sinusoidal_positional_encoding(src_lookup.shape[1], src_lookup.shape[-1])
    src_enc = add_positional_encoding_to_embeddings(scaled_embed, pe)

    scaled_embed = scale_embeddings_by_sqrt_d_model(tgt_lookup, tgt_lookup.shape[-1])
    pe = build_sinusoidal_positional_encoding(tgt_lookup.shape[1], tgt_lookup.shape[-1])
    tgt_enc = add_positional_encoding_to_embeddings(scaled_embed, pe)

    causal_mask = build_causal_mask(tgt_enc.shape[1])
    combined_tgt = combine_padding_and_causal_masks(tgt_mask, causal_mask)

    encoder_output = stack_encoder_layers(src_enc,model_params['encoder_layers'], num_heads, src_mask)
    decoder_output = stack_decoder_layers(tgt_enc, encoder_output, model_params['decoder_layers'], num_heads, src_mask, combined_tgt)

    logits = apply_final_output_projection(decoder_output,model_params['output_projection'])
    probabilities = apply_log_softmax_over_vocab(logits)
    
    return probabilities