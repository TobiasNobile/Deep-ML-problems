import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	for k in range(len(A)):

		for m in range(k+1, len(A)):
			if abs(A[k][k]) < abs(A[m][k]):
				A[k], A[m] = A[m].copy(), A[k].copy()
				b[k], b[m] = b[m].copy(), b[k].copy()
		
		for i in range(k+1, len(A)):
			m = A[i][k] / A[k][k]

			for j in range(k, A.shape[1]):
				A[i][j] = A[i][j] - m * A[k][j]
			b[i] = b[i] - m*b[k]

	for k in range(len(A)-1, -1, -1):
		for i in range(k+1, len(A)):
			b[k] = b[k] - A[k][i] * b[i]
		b[k] = b[k] / A[k][k]
	return b
