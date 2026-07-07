import numpy as np

def softmax(values):
	softed = []
	for i in range(len(values)):
		soft = np.exp(values[i])/np.sum(np.exp(values[i]))
		softed.append(soft)
	return softed
def pattern_weaver(n, crystal_values, dimension):
	scores_matrix = []
	for i in range(n):
		scores = []
		value_i = crystal_values[i]
		for j in range(n):
			scores.append(value_i*crystal_values[j]/np.sqrt(dimension))
		scores_matrix.append(scores)
	softed = softmax(scores_matrix)
	values = []
	for i in range(n):
		value = 0
		for j in range(n):
			value += softed[i][j]*crystal_values[j]
		values.append(value)

	return np.round(values,4).tolist()