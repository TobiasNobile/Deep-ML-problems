import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    distances = []
    query_array = np.array(list(query_point))
    for i in range(len(points)):
        p_array = np.array(list(points[i]))
        d = np.linalg.norm(query_array-p_array)
        distances.append((i, d))
    distances = sorted(distances, key=lambda x:x[1])[:k]

    res = []
    for d in distances:
        res.append(points[d[0]])
    return res
    
    