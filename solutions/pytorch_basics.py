import torch
import torch.nn as nn
import torch.nn.functional as F

def to_float_tensor(values):
    """
    Return a torch.float32 tensor built from `values`
    """
    return torch.tensor(values, dtype=torch.float32)

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    """
    Flatten x to 1-D, then rearrange into new_shape
    """
    return torch.reshape(torch.flatten(x), new_shape)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    """
    Swap the last two dimensions of x
    """
    return torch.transpose(x, -1, -2)

def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add b to every row of x using broadcasting

    How to check if dimensions are compatible for broadcasting ?
    1. Aligner la dernière dimension de chaque tenseur sur la droite
    2. Si elles sont égales ou si l'une d'elle vaut 1 ou est vide -> compatible
    """
    return x + b

def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Implement y = x W^T + b using PyTorch ops
    """
    return x @ W.T + b

def grad_of_quadratic(x_value: float) -> float:
    """
    Build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    """
    x_tensor = torch.tensor(x_value, dtype=torch.float32, requires_grad=True)
    forward = x_tensor**2 + 3*x_tensor + 2
    forward.backward()
    return x_tensor.grad.item()

class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Call the parent constructor and register an nn.Linear as self.linear
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the output of the linear layer
        """
        return self.linear(x) 
    
def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    """
    Build an SGD optimizer, run one full forward/loss/backward/step cycle,
    and return the pre-update loss as a Python float.
    """
    x.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    pred = model(x)
    loss = F.mse_loss(pred, y)
    loss.backward() # calcul des gradients
    optimizer.step() # mise à jour des poids
    return loss.item()

def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-15) -> float:
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability
    
    Returns:
        Mean binary cross-entropy loss
    """
    loss = torch.nn.BCELoss(reduction="mean")
    y_pred = torch.clip(y_pred, epsilon, 1-epsilon)
    return loss(y_pred, y_true).item()