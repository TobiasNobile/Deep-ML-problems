import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size)*0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
        self.W_hy = np.random.randn(output_size, hidden_size)*0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        h = np.zeros((self.hidden_size, 1))
        self.hs = [h]                 # hs[0] = état initial
        hidden_pred = []
        for i in range(len(x)):
            h = np.tanh(self.W_xh @ x[i].reshape(-1, 1) + self.W_hh @ h + self.b_h)
            y = self.W_hy @ h + self.b_y
            hidden_pred.append(y)
            self.hs.append(h)         # hs[t+1] = état après le pas t
        return hidden_pred

    def backward(self, x, y, learning_rate):
        dW_xh = np.zeros(self.W_xh.shape)
        dW_hh = np.zeros(self.W_hh.shape)
        dW_hy = np.zeros(self.W_hy.shape)
        db_h  = np.zeros(self.b_h.shape)
        db_y  = np.zeros(self.b_y.shape)

        hidden_pred = self.forward(x)
        dh_next = np.zeros((self.hidden_size, 1))
        for t in range(len(x)-1, -1, -1):
            h_t    = self.hs[t+1]     # état du pas t
            h_prev = self.hs[t]       # état du pas t-1
            dy = hidden_pred[t] - y[t].reshape(-1, 1)

            dW_hy += dy @ h_t.T
            db_y  += dy

            dh = self.W_hy.T @ dy + dh_next
            dh_raw = dh * (1 - h_t**2)         # dérivée tanh
            db_h  += dh_raw
            dW_xh += dh_raw @ x[t].reshape(-1, 1).T
            dW_hh += dh_raw @ h_prev.T
            dh_next = self.W_hh.T @ dh_raw      # gradient transmis au pas précédent

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h  -= learning_rate * db_h
        self.b_y  -= learning_rate * db_y