import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# note that we could improve far more this function with encapsuled fucntions
	# here it is just the basic algorithm
	updated_weights, updated_bias = initial_weights, initial_bias
	n = len(features)
	mse_values = []

	for i in range(epochs):
		pred = features.dot(updated_weights) + updated_bias
		pred = 1/(1+np.exp(-pred))
		error = pred - labels

		gradients = []
		for iWeight, w in enumerate(updated_weights):
			gradient = 0
			for j in range(n):
				gradient += error[j]*pred[j]*(1-pred[j])*features[j][iWeight]
			gradients.append(gradient*2/n)
		
		gradient_b = 2/n * sum(np.multiply(error, np.multiply(pred, 1-pred)))

		mse = 1/n * sum(error**2)
		mse_values.append(mse)

		gradients = np.array(gradients)
		updated_weights = updated_weights - learning_rate*gradients
		updated_bias = updated_bias - learning_rate*gradient_b


	return np.round(updated_weights, 4), np.round(updated_bias, 4), np.round(mse_values, 4)