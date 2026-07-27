import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters (He initialization).
        '''
        self.lr = lr

        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

        self.cache = {}

    def forward(self, X):
        '''
        Forward pass through the network.

        Args:
            X: Input batch, shape (N, 784)

        Returns:
            probs: Class probabilities, shape (N, 10)
        '''
        self.cache["X"]  = X
        self.cache["z1"] = X @ self.w1 + self.b1
        self.cache["h1"] = np.maximum(0, self.cache["z1"])
        self.cache["z2"] = self.cache["h1"] @ self.w2 + self.b2

        z = self.cache["z2"]
        softed = np.exp(z - np.max(z, axis=1, keepdims=True))
        return softed / np.sum(softed, axis=1, keepdims=True)

    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients and update all parameters.

        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities, shape (N, 10)

        Returns:
            loss: Scalar cross-entropy loss
        '''
        N = len(X)
        Y = np.eye(probs.shape[1])[y]                # (N,) -> (N, 10) one-hot

        loss = -np.sum(Y * np.log(probs + 1e-12)) / N

        d_z2   = (probs - Y) / N
        g_L_w2 = self.cache["h1"].T @ d_z2
        g_L_b2 = np.sum(d_z2, axis=0)
        g_L_h1 = d_z2 @ self.w2.T

        d_z1   = g_L_h1 * (self.cache["z1"] > 0)
        g_L_w1 = X.T @ d_z1
        g_L_b1 = np.sum(d_z1, axis=0)

        self.w2 -= self.lr * g_L_w2
        self.b2 -= self.lr * g_L_b2
        self.w1 -= self.lr * g_L_w1
        self.b1 -= self.lr * g_L_b1

        return loss

    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.

        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)

        Returns:
            loss: Scalar loss value
        '''
        probs = self.forward(X)
        return self.backward(X, y, probs)

    def predict(self, X):
        '''
        Predict class labels.

        Args:
            X: Input batch, shape (N, 784)

        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)