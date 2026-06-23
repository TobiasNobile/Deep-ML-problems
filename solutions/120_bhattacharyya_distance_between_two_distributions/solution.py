import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        return 0
    BC = np.sum(np.sqrt(np.multiply(p, q)))
    return -np.log(BC)