import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    
    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    match norm_type:
        case "l1":
            return np.sum(np.absolute(arr))
        case "l2":
            return np.sqrt(np.sum(arr**2))
        case "frobenius":
            return np.sqrt(np.sum(np.absolute(arr)**2))
    