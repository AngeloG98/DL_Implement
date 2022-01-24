import numpy as np

class Relu:
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        delta[self.x < 0] = 0
        return delta


class Softmax:
    def cal_loss(self, predict, label):
        batchsize, classes = predict.shape
        self.predict(predict)
        loss = 0
        delta = np.zeros(predict.shape)
        for i in range(batchsize):
            delta[i] = self.softmax[i] - label[i]
            loss -= np.sum(np.log(self.softmax[i]) * label[i])
        loss /= batchsize
        return loss, delta

    def predict(self, predict):
        batchsize, classes = predict.shape
        self.softmax = np.zeros(predict.shape)
        for i in range(batchsize):
            predict_tmp = predict[i] - np.max(predict[i])
            predict_tmp = np.exp(predict_tmp)
            self.softmax[i] = predict_tmp / np.sum(predict_tmp)
        return self.softmax

class Fcn:
    def __init__(self, size) -> None:
        scale = np.sqrt(size[0] / 2)
        self.weight = np.random.standard_normal((size[0], size[1])) / scale
        self.bias = np.random.standard_normal(size[1]) / scale
        self.weight_gradient = np.zeros((size[0], size[1]))
        self.biaw_gradient = np.zeros(size[1])

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, delta, lr):
        self.weight_gradient = np.dot(self.x.T, delta)
        self.biaw_gradient = np.sum(delta, axis=0)
        batch_size = delta.shape[0]
        self.weight -= lr / batch_size * self.weight_gradient
        self.bias -= lr / batch_size * self.biaw_gradient
        
        delta_backward = np.dot(delta, self.weight.T)

        return delta_backward

# Convolutional layer
class Conv:
    def __init__(self, kernel_size, stride = 1, pad = 0) -> None:
        """ Define a convolutional layer.
        1.kernel (w, h, d) * n :
            width(w)
            height(h)
            in_depth(d): depth(channel) of the input feature map
            out_depth(n): number of kernels, also depth of the out feature map
        2.stride and pad
        3.init kernel and bias : (w, h, d, n) and n
        4.init kernel and bias gradient, same size
        """
        width, height, in_depth, out_depth = kernel_size

        self.stride = stride 
        self.pad = pad

        scale = np.sqrt(3 * in_depth * width * height / out_depth)
        self.kernel = np.random.standard_normal(kernel_size) / scale
        self.bias = np.random.standard_normal(out_depth) / scale
        
        self.kernel_gradient = np.zeros(kernel_size)
        self.bias_gradient = np.zeros(out_depth)

    def img2col(self, xi):
        wx, hx, dx = xi.shape
        wk, hk, dk, nk = self.kernel.shape
        wfeature = (wx - wk)//self.stride + 1
        hfeature = (hx - hk)//self.stride + 1
        image_col = np.zeros((wfeature*hfeature, wk*hk*dk))
        for dxi in range(dx):
            one_channel = np.zeros((wfeature*hfeature, wk*hk))
            idx = 0
            for i in range(wfeature):
                for j in range(hfeature):
                    one_channel[idx] = xi[i*self.stride:i*self.stride + wk, j*self.stride:j*self.stride + hk, dxi].reshape(-1)
                    idx += 1
            image_col[:, dxi*wk*hk:(dxi+1)*wk*hk] = one_channel
        return image_col

    def forward(self, x):
        self.x = x
        if self.pad != 0:
            self.x = np.pad(self.x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant')
        bx, wx, hx, dx = self.x.shape
        wk, hk, dk, nk = self.kernel.shape
        self.x_col_list = []
        kernel_col = self.kernel.reshape(-1, nk)

        wfeature = (wx - wk)//self.stride + 1
        hfeature = (hx - hk)//self.stride + 1
        feature = np.zeros((bx, wfeature, hfeature, nk))

        for i in range(bx):
            x_col = self.img2col(x[i])
            feature[i] = (np.dot(x_col, kernel_col) + self.bias).reshape(wfeature, hfeature, nk)
            self.x_col_list.append(x_col)
        
        return feature
        
    def backward(self, delta, lr):
        bd, wd, hd, dd = delta.shape
        bx, wx, hx, dx = self.x.shape
        wk, hk, dk, nk = self.kernel.shape

        delta_col = delta.reshape(bd, -1, dd)
        for i in range(bx): 
            self.kernel_gradient += np.dot(self.x_col_list[i], delta_col[i]).reshape(self.kernel_gradient.shape)
        self.kernel_gradient /= bx
        self.bias_gradient += np.sum(delta_col, axis=(0,1))
        self.bias_gradient /= bx

        self.kernel -= self.kernel_gradient * lr
        self.bias -= self.bias_gradient * lr

        delta_backward = np.zeros(self.x.shape)
        kernel_rot = np.rot90(self.kernel, 2)
        kernel_rot_col = kernel_rot.reshape(-1, nk)

        if hd - hk + 1 != hx:
            pad = (hx - hd + hk - 1) // 2
            pad_delta = np.pad(delta, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        else:
            pad_delta = delta

        for i in range(bx):
            pad_delta_col = self.img2col(pad_delta[i])
            delta_backward[i] = np.dot(pad_delta_col, kernel_rot_col).reshape(wx, hx, dk)

        return delta_backward

class Pool:
    def forward(self, x):
        x = x
        bx, wx, hx, dx = x.shape
        feature = np.zeros((bx, wx//2, hx//2, dx))
        self.feature_mask = np.zeros((bx, wx, hx, dx))
        for b in range(bx):
            for d in range(dx):
                for i in range(wx//2):
                    for j in range(hx//2):
                        feature[b, i, j, d] = np.max(x[b, 2*i:2*i+2, 2*j:2*j+2, d])
                        index = np.argmax(x[b, 2*i:2*i+2, 2*j:2*j+2, d])
                        self.feature_mask[b, i * 2 + index // 2, j * 2 + index % 2, d] = 1
        return feature

    def backward(self, delta):
        np.repeat(np.repeat(delta, 2, axis=1), 2, axis=2) * self.feature_mask


class CNN:
    def __init__(self) -> None:
        """
        28x28x1------
                    | conv1 (5,5,1) x 6
        24x24x6------
                    | max pool1 
        12x12x6------
                    | conv1 (5,5,6) x 16
        8x8x16-------
                    | max pool2 
        4x4x16-------
                    | flatten-> fcn 256 x 10-> softmax
        10-----------
        """
        self.conv1 = Conv(kernel_size=(5, 5, 1, 6))
        self.relu1 = Relu()
        self.pool1 = Pool() 
        self.conv2 = Conv(kernel_size=(5, 5, 6, 16))
        self.relu1 = Relu()
        self.pool2 = Pool() 
        self.fcn = Fcn(256, 10)
        self.softmax = Softmax()
    
    def train(self, train_X, train_Y, batchsize=32, lr=0.01, epochs=10):
        for epoch in range(epochs):
            for i in range(0, train_X.shape[0], batchsize):
                X = train_X[i:i + batchsize]
                Y = train_Y[i:i + batchsize]

                predict = self.conv1.forward(X)
                predict = self.relu1.forward(predict)
                predict = self.pool1.forward(predict)
                predict = self.conv2.forward(predict)
                predict = self.relu2.forward(predict)
                predict = self.pool2.forward(predict)
                predict = predict.reshape(batchsize, -1)
                predict = self.nn.forward(predict)


                loss, delta = self.softmax.cal_loss(predict, Y)

                delta = self.nn.backward(delta, lr)
                delta = delta.reshape(batchsize, 4, 4, 16)
                delta = self.pool2.backward(delta)
                delta = self.relu2.backward(delta)
                delta = self.conv2.backward(delta, lr)
                delta = self.pool1.backward(delta)
                delta = self.relu1.backward(delta)
                self.conv1.backward(delta, lr)

                print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

            lr *= 0.95 ** (epoch + 1)
    
    def save(self):
        np.savez("simple_cnn_model.npz", \
        k1=self.conv1.kernel, b1=self.conv1.bias, k2=self.conv2.kernel, b2=self.conv2.bias, w3=self.fcn.weight, b3=self.fcn.bias)