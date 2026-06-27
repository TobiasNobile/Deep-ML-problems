from encoder_decoder_full_model import *
from training_objective_and_schedule import *

def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    """
    Shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    """
    pad_id, start_id, vocab_size = config["pad_id"], config["start_id"], config["vocab_size"]
    smoothing, num_heads = config["smoothing"], config["num_heads"]

    shifted = shift_targets_right_with_start_token(tgt_batch, start_id)
    proba = run_transformer_forward(src_batch, shifted, model_params, num_heads, pad_id)

    uniform_distrb = build_uniform_smoothing_distribution(proba.shape, vocab_size, smoothing)
    confidence_gold = set_confidence_on_gold_tokens(uniform_distrb, tgt_batch, 1-smoothing)
    distrib = zero_pad_column_and_pad_token_rows(confidence_gold, tgt_batch, pad_id)
    kl_loss = compute_label_smoothed_kl_loss(proba, distrib)
    avg_loss = average_loss_over_non_pad_tokens(kl_loss, tgt_batch, pad_id)
    return avg_loss

