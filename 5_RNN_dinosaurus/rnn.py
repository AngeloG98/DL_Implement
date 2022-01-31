import numpy as np

class RNN:
    def __init__(self, hidden_size, in_size, out_size) -> None:

        self.hidden_init = np.zeros((hidden_size, 1))

        self.W_xh = np.random.randn((hidden_size, in_size)) * 0.01
        self.W_hh = np.random.randn((hidden_size, hidden_size)) * 0.01
        self.W_hy = np.random.randn((out_size, hidden_size)) * 0.01
        self.b_h = np.random.randn((hidden_size, 1)) * 0.01
        self.b_y = np.random.randn((out_size, 1)) * 0.01

        self.dW_xh = np.zeros(self.W_xh.shape)
        self.dW_hh = np.zeros(self.W_hh.shape)
        self.dW_hy = np.zeros(self.W_hy.shape)
        self.db_h = np.zeros(self.b_h.shape)
        self.db_y = np.zeros(self.b_y.shape)
        
        self.in_size = in_size
        self.out_size = out_size

    def softmax(self, x):
        x_tmp = x - np.max(x) # for numerical stability
        x_tmp = np.exp(x_tmp)
        x_softmax = x_tmp / np.sum(x_tmp)
        return x_softmax

    def forward(self, input, target):
        # resest
        self.x_seq, self.h_seq, self.y_pred, self.p_pred = {}, {}, {}, {} # cause of [-1], use dict here
        self.h_seq[-1] = self.hidden_init
        self.loss = 0

        for t in range(input.shape[0]):
            self.x_seq[t] = np.zeros((self.in_size, 1))
            self.x_seq[t][input[t]] = 1 # one-hot
            self.h_seq[t] = np.tanh(np.dot(self.W_xh, self.x_seq[t]) + np.dot(self.W_hh, self.h_seq[t-1]) + self.b_h)
            self.y_pred[t] = np.dot(self.W_hy, self.h_seq[t]) + self.b_y
            self.p_pred[t] = self.softmax(self.y_pred[t])

            y_label = np.zeros((self.out_size, 1))
            y_label[target[t]] = 1
            self.loss += np.sum(-np.log(self.p_pred[t]) * y_label) # cross-entropy loss
        
    def backward(self, input):
        # reset
        self.dW_xh = np.zeros(self.W_xh.shape)
        self.dW_hh = np.zeros(self.W_hh.shape)
        self.dW_hy = np.zeros(self.W_hy.shape)
        self.db_h = np.zeros(self.b_h.shape)
        self.db_y = np.zeros(self.b_y.shape)

        for t in reversed(range(input.shape[0])):
            pass

        # backprop through layer

        # backprop through time

