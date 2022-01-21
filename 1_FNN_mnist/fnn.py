import numpy as np

class FNN():
    def __init__(self, size) -> None:
        # for example: size = [784,30,10]
        self.num_layers = len(size)
        self.weights = [np.random.randn(x,y) for x,y in zip(size[1:],size[:-1])]
        self.biases = [np.random.randn(x,1) for x in size[1:]]
        self.loss_list = []
    
    def activate(self, z):
        #sigmoid
        f_z = 1.0/(1.0 + np.exp(-z))
        return f_z

    def activate_d(self, z):
        #sigmoid
        fd_z = self.activate(z)*(1 - self.activate(z))
        return fd_z
    
    def loss(self, y, y_hat):
        y = y.flatten()
        y_hat = y_hat.flatten()
        return 1/2*((y-y_hat).T.dot(y-y_hat))
    
    def loss_d(self, y, y_hat):
        return -(y - y_hat)

    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activate(z)
        return a
    
    def back_propagation(self, x, y):
        #do feed forward and storage data
        a = x
        a_list = [x]
        z_list = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            z_list.append(z)
            a = self.activate(z)
            a_list.append(a)
        
        nabla_w = [np.zeros(wi.shape) for wi in self.weights]
        nabla_b = [np.zeros(bi.shape) for bi in self.biases]
        #do back-propagation
        kkk= self.loss_d(y, a_list[-1])
        delta = self.loss_d(y, a_list[-1])*self.activate_d(z_list[-1])
        nabla_w[-1] = np.dot(delta,a_list[-2].T)
        nabla_b[-1] = delta
        for L in range(2, self.num_layers):
            delta = np.dot(self.weights[-L+1].T,delta)*self.activate_d(z[-L])
            nabla_w[-L] = np.dot(delta,a_list[-L-1].T)
            nabla_b[-L] = delta
        return nabla_w, nabla_b

    def update(self, mini_batch, batchsize, alpha):
        nabla_w = [np.zeros(wi.shape) for wi in self.weights]
        nabla_b = [np.zeros(bi.shape) for bi in self.biases]
        for x,y in mini_batch:
            nabla_w_i, nabla_b_i = self.back_propagation(x, y)
            nabla_w = [nw+nw_i for nw,nw_i in zip(nabla_w, nabla_w_i)]
            nabla_b = [nb+nb_i for nb,nb_i in zip(nabla_b, nabla_b_i)]
        #gradient descent
        self.weights = [w - alpha*nw/batchsize for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b - alpha*nb/batchsize for b,nb in zip(self.biases, nabla_b)]
        loss_i = 0
        for x,y in mini_batch:
            loss_i += self.loss(y, self.feed_forward(x))
        self.loss_list.append(loss_i/batchsize)
        
    def train(self, traning_data, alpha, batchsize, epochs, test_data):
        data_len = len(traning_data)
        for i in range(epochs):
            np.random.shuffle(traning_data)
            mini_batches = [ traning_data[k:k + batchsize] for k in range(0, data_len, batchsize)]
            k = 0
            for mini_batch in mini_batches:
                self.update(mini_batch, batchsize, alpha)
                if k%50 == 0:
                    print("epoch:{}, mini_batch:{}, loss:{}, accuracy rate:{}".format(i,k,self.loss_list[-1],self.predict(test_data)/len(test_data)))
                k += 1
                

    def predict(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    


    
