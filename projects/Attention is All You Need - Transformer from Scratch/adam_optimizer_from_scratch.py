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

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    """
    Increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    """
    optimizer_state["t"] += 1
    for i in range(len(parameter_list)):
        if parameter_list[i].grad is None:
            continue
        g = parameter_list[i].grad 
        optimizer_state["m"][i] = beta1*optimizer_state["m"][i] + (1 - beta1)*g
        optimizer_state["v"][i] = beta2*optimizer_state["v"][i] + (1 - beta2)*g**2

        m_hat = optimizer_state["m"][i]/(1 - beta1**optimizer_state["t"])
        v_hat = optimizer_state["v"][i]/(1 - beta2**optimizer_state["t"])

        parameter_list[i].data = parameter_list[i].data - learning_rate * m_hat/(torch.sqrt(v_hat) + epsilon)

    return optimizer_state 

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    for i in range(len(parameter_list)):
        parameter_list[i].grad = None