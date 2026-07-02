import numpy as np

def divide_on_feature(X, feature_i, threshold):
	X_over, X_under = [], []
	for i in range(len(X)):
		if X[i][feature_i] >= threshold:
			X_over.append(X[i])
		else:
			X_under.append(X[i])
	return [X_over, X_under]
