import numpy as np

def cramers_rule(A, b):
    A, b = np.array(A), np.array(b)
    if np.linalg.det(A) == 0:
        return -1
    x = []
    for i in range(len(A)):
        A = A.T
        A_modified = A.copy()
        A_modified[i] = b.copy()
        A_modified.T
        x_i = np.linalg.det(A_modified)/np.linalg.det(A)
        x.append(x_i)

        A = A.T
    return x
