import numpy as np

class RNN:
    def __init__(self, hidden_size, in_size, out_size) -> None:

        self.hidden_init = np.zeros((hidden_size, 1))

        self.W_xh = np.random.randn(hidden_size, in_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(out_size, hidden_size) * 0.01
        self.b_h = np.random.randn(hidden_size, 1) * 0.01
        self.b_y = np.random.randn(out_size, 1) * 0.01

        self.dW_xh = np.zeros(self.W_xh.shape)
        self.dW_hh = np.zeros(self.W_hh.shape)
        self.dW_hy = np.zeros(self.W_hy.shape)
        self.db_h = np.zeros(self.b_h.shape)
        self.db_y = np.zeros(self.b_y.shape)
        
        self.in_size = in_size
        self.out_size = out_size

    def zero_grad(self):
        self.dW_xh_b = np.zeros(self.W_xh.shape)
        self.dW_hh_b = np.zeros(self.W_hh.shape)
        self.dW_hy_b = np.zeros(self.W_hy.shape)
        self.db_h_b = np.zeros(self.b_h.shape)
        self.db_y_b = np.zeros(self.b_y.shape)

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

        for t in range(len(input)):
            self.x_seq[t] = np.zeros((self.in_size, 1))
            self.x_seq[t][input[t]] = 1 # one-hot
            self.h_seq[t] = np.tanh(np.dot(self.W_xh, self.x_seq[t]) + np.dot(self.W_hh, self.h_seq[t-1]) + self.b_h)
            self.y_pred[t] = np.dot(self.W_hy, self.h_seq[t]) + self.b_y
            self.p_pred[t] = self.softmax(self.y_pred[t])

            y_label = np.zeros((self.out_size, 1))
            y_label[target[t]] = 1
            self.loss += np.sum(-np.log(self.p_pred[t]) * y_label) # cross-entropy loss
        
    def backward(self, input, target):
        # reset
        self.dW_xh = np.zeros(self.W_xh.shape)
        self.dW_hh = np.zeros(self.W_hh.shape)
        self.dW_hy = np.zeros(self.W_hy.shape)
        self.db_h = np.zeros(self.b_h.shape)
        self.db_y = np.zeros(self.b_y.shape)
        # dL_ht = np.zeros(self.hidden_init.shape)
        dhnext = np.zeros(self.hidden_init.shape)

        for t in reversed(range(len(input))):
            # backprop through layer
            dLt_yt = np.copy(self.p_pred[t])
            dLt_yt[target[t]] -= 1
            self.dW_hy += np.dot(dLt_yt, self.h_seq[t].T)
            self.db_y += dLt_yt

            # backprop through time 1
            # dLt_ht = np.dot(self.W_hy.T, dLt_yt)
            # if t == len(input)-1:
            #     dL_ht = dLt_ht
            # else:
            #     dhnext_h = np.dot(self.W_hh.T, (1 - self.h_seq[t+1] * self.h_seq[t+1]))
            #     dL_ht = dLt_ht + dhnext_h * dL_ht
            # dL_zt = (1 - self.h_seq[t] * self.h_seq[t]) * dL_ht
            # self.dW_hh += np.dot(dL_zt, self.h_seq[t].T)
            # self.dW_xh += np.dot(dL_zt, self.x_seq[t].T)
            # self.db_h += dL_zt

            # backprop through time 2
            dh = np.dot(self.W_hy.T, dLt_yt) + dhnext # backprop into h. 
            dhraw = (1 - self.h_seq[t] * self.h_seq[t]) * dh # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
            self.db_h += dhraw
            self.dW_xh += np.dot(dhraw, self.x_seq[t].T)
            self.dW_hh += np.dot(dhraw, self.h_seq[t-1].T)
            dhnext = np.dot(self.W_hh.T, dhraw)
        for dparam in [self.dW_xh, self.dW_hh, self.dW_hy, self.db_h, self.db_y]: 
            np.clip(dparam, -5, 5, out=dparam) # gradient clip

    def train(self, train_data, ix_to_char, char_to_ix, batchsize=2, lr=0.1, epochs=30):
        mW_xh = np.zeros(self.W_xh.shape)
        mW_hh = np.zeros(self.W_hh.shape)
        mW_hy = np.zeros(self.W_hy.shape)
        mb_h = np.zeros(self.b_h.shape)
        mb_y = np.zeros(self.b_y.shape)
        for epoch in range(epochs):
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), batchsize):
                self.zero_grad()

                batch_data = train_data[i:i + batchsize]
                for data in batch_data:
                    X = [None] + [char_to_ix[ch] for ch in data]
                    Y = X[1:] + [char_to_ix["\n"]]
                    self.forward(X, Y)
                    self.backward(X, Y)
                    self.dW_xh_b += self.dW_xh
                    self.dW_hh_b += self.dW_hh
                    self.dW_hy_b += self.dW_hy
                    self.db_h_b += self.db_h
                    self.db_y_b += self.db_y

                k = len(batch_data)
                for param, dparam,mem in zip([self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y],
                                        [self.dW_xh_b, self.dW_hh_b, self.dW_hy_b, self.db_h_b, self.db_y_b],
                                        [mW_xh, mW_hh, mW_hy, mb_h, mb_y]):
                    mem += dparam * dparam
                    param += -lr/k * dparam / np.sqrt(mem + 1e-8)
            
            # print each epoch        
            print("epoch: {}, loss: {:.4f}".format(epoch, self.loss))
            seed = 0
            for i in range(5):
                print("predict: ", self.predict(char_to_ix, ix_to_char, seed)[1])
                seed += 1

        
    def predict(self, char_to_ix, ix_to_char, seed):
        x = np.zeros((self.in_size, 1))
        h_prev = np.zeros(self.hidden_init.shape)
        
        indices = []
        idx = -1 
        counter = 0
        newline_character = char_to_ix['\n']
        
        while (idx != newline_character and counter != 50):
            # forward
            h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, h_prev) + self.b_h)
            y = self.softmax(np.dot(self.W_hy, h) + self.b_y)
            # sample
            np.random.seed(counter + seed) 
            idx = np.random.choice(range(len(y)),p=y.ravel())
            indices.append(idx)
            # update x h
            x = np.zeros((self.in_size, 1))
            x[idx] = 1
            h_prev = h
            # update seed
            seed += 1
            counter +=1
        if (counter == 50):
            indices.append(char_to_ix['\n'])
        
        indices_char = ''
        for i in indices:
            indices_char += ix_to_char[i]

        return indices, indices_char[:-1]

    

