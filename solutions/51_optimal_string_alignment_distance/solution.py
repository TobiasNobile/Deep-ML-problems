import torch

def OSA(source: str, target: str) -> torch.Tensor:
    """
    Calculate the Optimal String Alignment (OSA) distance between two strings.
    Uses PyTorch tensors to store the dynamic programming matrix.
    """
    m, n = len(source), len(target)
    d = torch.zeros(m + 1, n + 1)

    for i in range(m + 1):
        d[i, 0] = i
    for j in range(n + 1):
        d[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            
            d[i, j] = min(
                d[i - 1, j] + 1,      
                d[i, j - 1] + 1,     
                d[i - 1, j - 1] + cost 
            )
            
            if i > 1 and j > 1 and source[i - 1] == target[j - 2] and source[i - 2] == target[j - 1]:
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + 1) 
                
    return d[m, n]