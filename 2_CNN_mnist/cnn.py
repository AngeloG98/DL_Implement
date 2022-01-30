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
            loss -= np.sum(np.log(self.softmax[i]) * label[i]) #nll loss
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
        """ Define a fully-connected layer.
        size: (input, output)
        """
        scale = np.sqrt(size[0] / 2)
        self.weight = np.random.standard_normal((size[0], size[1])) / scale
        self.bias = np.random.standard_normal(size[1]) / scale
        self.weight_gradient = np.zeros((size[0], size[1]))
        self.bias_gradient = np.zeros(size[1])

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, delta, lr):
        # backprop first
        batch_size = delta.shape[0]
        delta_backward = np.dot(delta, self.weight.T)
        # update weight and bias
        self.weight_gradient = np.dot(self.x.T, delta)
        self.bias_gradient = np.sum(delta, axis=0)
        self.weight -= lr / batch_size * self.weight_gradient
        self.bias -= lr / batch_size * self.bias_gradient
        return delta_backward

# Convolutional layer
class Conv:
    def __init__(self, kernel_size, stride = 1, pad = 0) -> None:
        """ Define a convolutional layer.
        1.kernel n * d * (w, h)  :
            out_depth(n): number of kernels, also depth of the out feature map
            in_depth(d): depth(channel) of the input feature map
            width(w)
            height(h)
        2.stride and pad
        3.init kernel and bias : (n, d, w, h) and n
        4.init kernel and bias gradient, same size
        """
        out_depth, in_depth, width, height = kernel_size

        self.stride = stride 
        self.pad = pad

        scale = np.sqrt(3 * in_depth * width * height / out_depth)
        self.kernel = np.random.standard_normal(kernel_size) / scale
        self.bias = np.random.standard_normal(out_depth) / scale
        
        self.kernel_gradient = np.zeros(kernel_size)
        self.bias_gradient = np.zeros(out_depth)

    def img2col(self, xi):
        dx, wx, hx = xi.shape
        nk, dk, wk, hk = self.kernel.shape
        wfeature = (wx - wk)//self.stride + 1
        hfeature = (hx - hk)//self.stride + 1
        image_col = np.zeros((wfeature*hfeature, wk*hk*dx))
        idx = 0
        for i in range(wfeature):
            for j in range(hfeature):
                image_col[idx] = xi[:, i*self.stride:i*self.stride + wk, j*self.stride:j*self.stride + hk].reshape(-1)
                idx += 1
        return image_col

    def forward(self, x):
        self.x = x
        bx, dx, wx, hx = self.x.shape
        nk, dk, wk, hk = self.kernel.shape

        self.x_col_list = []
        kernel_col = self.kernel.reshape(nk, -1).T # transpose here !

        wfeature = (wx - wk)//self.stride + 1
        hfeature = (hx - hk)//self.stride + 1
        feature = np.zeros((bx, nk, wfeature, hfeature))

        for i in range(bx):
            x_col = self.img2col(x[i])
            feature_i = (np.dot(x_col, kernel_col) + self.bias).T # transpose first !
            feature[i] = feature_i.reshape(nk, wfeature, hfeature) # reshape
            self.x_col_list.append(x_col)
        
        return feature
        
    def backward(self, delta, lr):
        bd, dd, wd, hd = delta.shape
        bx, dx, wx, hx = self.x.shape
        nk, dk, wk, hk = self.kernel.shape
        # set gradient zero !!!
        self.kernel_gradient = np.zeros(self.kernel_gradient.shape)
        self.bias_gradient = np.zeros(self.bias_gradient.shape)
        delta_col = delta.reshape(bd, dd, -1).transpose((0,2,1)) # transpose: delta_depth <-> delta_w*delta_h
        for i in range(bx): 
            # explain: self.x_col_list[i].T
            # x_col is img2col according to kernel size
            # x_col.T is img2col according to delta(feature map) size !
            self.kernel_gradient += np.dot(self.x_col_list[i].T, delta_col[i]).T.reshape(self.kernel_gradient.shape) # transpose first ! -> reshape
        self.kernel_gradient /= bx
        self.bias_gradient += np.sum(delta_col, axis=(0,1))
        self.bias_gradient /= bx

        delta_backward = np.zeros(self.x.shape)
        # rot 180, then swapaxes -> reshape to match delta col !
        kernel_rot = np.rot90(self.kernel, 2, (2,3))
        kernel_rot_swap = kernel_rot.swapaxes(0,1)
        kernel_rot_col = kernel_rot_swap.reshape(dk,-1).T
        kernel_rot_col = kernel_rot.reshape(-1, dk) 

        if hd - hk + 1 != hx:
            pad = (hx - hd + hk - 1) // 2
            pad_delta = np.pad(delta, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
        else:
            pad_delta = delta

        for i in range(bx):
            pad_delta_col = self.img2col(pad_delta[i])
            delta_backward[i] = np.dot(pad_delta_col, kernel_rot_col).T.reshape(dk, wx, hx) # transpose first ! -> reshape

        #update kernel and bias
        self.kernel -= self.kernel_gradient * lr
        self.bias -= self.bias_gradient * lr

        return delta_backward

class Pool:
    def forward(self, x):
        x = x
        bx, dx, wx, hx = x.shape
        feature = np.zeros((bx, dx, wx//2, hx//2))
        self.feature_mask = np.zeros((bx, dx, wx, hx))
        for b in range(bx):
            for d in range(dx):
                for i in range(wx//2):
                    for j in range(hx//2):
                        feature[b, d, i, j] = np.max(x[b, d, 2*i:2*i+2, 2*j:2*j+2])
                        index = np.argmax(x[b, d, 2*i:2*i+2, 2*j:2*j+2])
                        self.feature_mask[b, d, i * 2 + index // 2, j * 2 + index % 2] = 1
        return feature

    def backward(self, delta):
        delta_backward = np.repeat(np.repeat(delta, 2, axis=3), 2, axis=2) * self.feature_mask
        return delta_backward


class CNN:
    def __init__(self) -> None:
        """
        28x28x1------
                    | conv1 (5,5,1) x 6
        24x24x6------
                    | max pool1 
        12x12x6------
                    | conv2 (5,5,6) x 16
        8x8x16-------
                    | max pool2 
        4x4x16-------
                    | flatten-> fcn 256 x 10-> softmax
        10-----------
        """
        self.conv1 = Conv(kernel_size=(6, 1, 5, 5))
        self.relu1 = Relu()
        self.pool1 = Pool() 
        self.conv2 = Conv(kernel_size=(16, 6, 5, 5))
        self.relu2 = Relu()
        self.pool2 = Pool() 
        self.fcn = Fcn((256, 10))
        self.softmax = Softmax()
    
    def train(self, train_X, train_Y, valid_X, valid_Y, batchsize=32, lr=0.01, epochs=3):
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
                predict = predict.reshape(X.shape[0], -1)
                predict = self.fcn.forward(predict)


                loss, delta = self.softmax.cal_loss(predict, Y)
                delta = self.fcn.backward(delta, lr)
                delta = delta.reshape(X.shape[0], 16, 4, 4)
                delta = self.pool2.backward(delta)
                delta = self.relu2.backward(delta)
                delta = self.conv2.backward(delta, lr)
                delta = self.pool1.backward(delta)
                delta = self.relu1.backward(delta)
                self.conv1.backward(delta, lr)

                print("epoch: {}, batch: {}, batchsize: {}, loss: {:.4f}".format(epoch, i, X.shape[0], loss))
                
                if i%11200 == 0 and i != 0:
                    self.save(epoch)
                if i%128 == 0 and i != 0:
                    state = np.random.get_state()
                    np.random.shuffle(valid_X)
                    np.random.set_state(state)
                    np.random.shuffle(valid_Y)
                    self.eval(valid_X[:50], valid_Y[:50])

            # lr *= 0.95 ** (epoch + 1)
            # print("change learning rate :{}".format(lr))
            self.save(epoch)
    
    def save(self, epoch):
        np.savez("2_CNN_mnist/model/mnist_cnn_model_"+str(epoch)+".npz", \
        k1=self.conv1.kernel, b1=self.conv1.bias, k2=self.conv2.kernel, b2=self.conv2.bias, w3=self.fcn.weight, b3=self.fcn.bias)
    
    def eval(self, test_X, test_Y, pre_train = False, filename = None):
        if pre_train == True:
            model = np.load(filename)
            self.conv1.kernel = model["k1"]
            self.conv1.bias = model["b1"]
            self.conv2.kernel = model["k2"]
            self.conv2.bias = model["b2"]
            self.fcn.weight = model["w3"]
            self.fcn.bias = model["b3"]

        test_size = test_X.shape[0]
        num = 0

        for i in range(test_size):
            X = test_X[i]
            X = X[:, np.newaxis]
            Y = test_Y[i]

            predict = self.conv1.forward(X)
            predict = self.relu1.forward(predict)
            predict = self.pool1.forward(predict)
            predict = self.conv2.forward(predict)
            predict = self.relu2.forward(predict)
            predict = self.pool2.forward(predict)
            predict = predict.reshape(1, -1)
            predict = self.fcn.forward(predict)
            predict = self.softmax.predict(predict)

            if np.argmax(predict) == Y:
                num += 1
            if i%1000 == 0 and i!=0:
                print("Processing, {} done...".format(i))

        print("accuracy rate: {}%".format(num / test_size * 100))