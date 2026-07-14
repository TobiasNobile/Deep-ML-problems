import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
	x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
	r = b - A @ x
	p = r.copy()

	for k in range(n):
		alpha = r.T @ r / (p.T @ A @ p)
		x += alpha * p
		r_pred = r.copy()
		r -= alpha * A @ p

		if np.linalg.norm(r) < tol:
			break
		
		beta = r.T @ r / (r_pred.T @ r_pred)
		p = r + beta * p
	return x
