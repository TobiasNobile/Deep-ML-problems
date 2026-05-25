import numpy as np

def pairwise_cosine_similarity(X):
    cosine_matrix = []
    for row in X:
        if np.linalg.norm(np.linalg.norm(row)) == 0:
            cosine_matrix.append([0]*len(row))
        else:
            cosine_matrix.append(row/np.linalg.norm(row))
    cosine_matrix = np.array(cosine_matrix)
    return cosine_matrix.dot(cosine_matrix.T)