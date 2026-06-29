import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    vectors = np.array(vectors)
    
    u = []
    for v in vectors:
        proj = np.sum([(v @ u_i) * u_i for u_i in u], axis=0)
        w_k = v - proj
        if np.linalg.norm(w_k) > tol:
            u_k = w_k/np.linalg.norm(w_k)
            u.append(u_k)
    return u