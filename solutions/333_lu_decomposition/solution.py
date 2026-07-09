import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	A = np.array(A)
	L, U = np.eye(len(A)), np.zeros((len(A), len(A)))
	for i in range(len(A)):
		U[i] = A[i] - L[i, :i] @ U[:i]
		L[:, i] = 1/U[i][i]*(A[:, i] - L[:, :i] @ U[:i, i])
	return L, U
    