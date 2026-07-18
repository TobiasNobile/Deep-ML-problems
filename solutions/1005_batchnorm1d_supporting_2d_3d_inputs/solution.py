import numpy as np

def batchnorm1d(x, gamma, beta, running_mean, running_var, training=True, momentum=0.1, eps=1e-5):
    """
    BatchNorm1d supporting 2D (N, C) and 3D (N, L, C) inputs.

    Returns a dict with keys 'out', 'running_mean', 'running_var'.
    """
    if training:
        if len(x.shape) ==2:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
        else:
            mu = np.mean(x, axis=(0, 1))
            var = np.var(x, axis=(0, 1))
    else:
        mu = running_mean
        var = running_var
    x_hat = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    running_mean = (1 - momentum) * running_mean + momentum * mu
    running_var = (1 - momentum) * running_var + momentum * var
    return {
        "out":out.tolist(),
        "running_mean": [float(m) for m in running_mean],
        "running_var": [float(v) for v in running_var]
    }
    
