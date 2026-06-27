import numpy as np

def elbow_wcss(X: np.ndarray, k_values: list, max_iters: int = 100) -> list:
    """
    Compute WCSS (inertia) for each k in k_values using K-Means.

    Args:
        X: Data of shape (n_samples, n_features)
        k_values: List of cluster counts to evaluate
        max_iters: Maximum number of Lloyd iterations

    Returns:
        List of WCSS values (rounded to 4 decimals), one per k
    """
    WCSS_list = []
    for k in k_values:
        WCSS = 0
        centroids = X[:k]
        for i in range(max_iters):
            clusters = {c:[] for c in range(len(centroids))}
            for j in range(len(X)):
                distances = [sum(abs(X[j] - centroids[c]) **2) for c in clusters]
                c_nearest = np.argmin((np.array(distances)))
                clusters[c_nearest].append(X[j])
            centroids = [np.mean(clusters[c], axis=0) for c in clusters]
        
        for c in clusters:
            for value in clusters[c]:
                WCSS += sum(abs(value - sum(clusters[c])/len(clusters[c]))**2)
        WCSS_list.append(WCSS)
    return [WCSS.tolist() for WCSS in WCSS_list]
            
