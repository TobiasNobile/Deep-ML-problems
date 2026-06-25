import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    state = {
        "m": [],
        "v": [],
        "t": 0
    }
    for param in parameter_list:
        state["m"].append(torch.zeros_like(param))
        state["v"].append(torch.zeros_like(param))
    return state

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    return beta1 * m_prev + (1 - beta1) * grad

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    return beta2 * v_prev + (1 - beta2) * grad ** 2

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    return m_t / (1 - beta1**step), v_t / (1 - beta2**step)