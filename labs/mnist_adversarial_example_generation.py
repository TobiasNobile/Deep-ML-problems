import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    '''
    Generate an adversarial example for input x.
    
    Args:
        model: Pre-trained classifier (already in eval mode)
        x: Input image tensor, shape (1, 1, 28, 28), values in [0,1]
        y: True label, shape (1,) or scalar
        epsilon: L∞ perturbation budget
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
    
    Returns:
        x_adv: Adversarial example, same shape as x, satisfying:
               - ||x_adv - x||_∞ ≤ epsilon
               - x_adv values in [0, 1]
               - model(x_adv).argmax() != y (ideally)
    '''
    # Enable gradient tracking on input
    x.requires_grad = True

    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward to get gradient w.r.t. x
    model.zero_grad()
    loss.backward()

    # Get sign of gradient
    grad_sign = x.grad.sign()

    # Perturb input
    x_adv = x + epsilon * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv
