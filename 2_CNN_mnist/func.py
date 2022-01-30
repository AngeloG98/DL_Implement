import numpy as np
class test:
    def __init__(self) -> None:
        # self.kernel = np.arange(1,55).reshape(3,2,3,3)
        # self.kernel = np.ones((3,2,3,3))
        # self.kernel = np.ones((3,2,2,2))
        self.kernel = np.arange(1,25).reshape(3,2,2,2)
        self.kernel_gradient = np.zeros(self.kernel.shape)
        self.stride = 1
        self.bias = np.zeros(3)
    
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
        kernel_col = self.kernel.reshape(nk, -1).T

        wfeature = (wx - wk)//self.stride + 1
        hfeature = (hx - hk)//self.stride + 1
        feature = np.zeros((bx, nk, wfeature, hfeature))

        for i in range(bx):
            x_col = self.img2col(x[i])
            feature_i = (np.dot(x_col, kernel_col) + self.bias).T
            feature[i] = feature_i.reshape(nk, wfeature, hfeature)
            self.x_col_list.append(x_col)
        
        return feature
    
    def backward(self, delta, x):
        bd, dd, wd, hd = delta.shape
        bx, dx, wx, hx = x.shape
        nk, dk, wk, hk = self.kernel.shape
        delta_col = delta.reshape(bd,dd, -1 )
        delta_col = delta_col.transpose((0,2,1))
        print("delta_col\n",delta_col)
        b_grad = np.sum(delta_col, axis=(0,1))
        # print("b_grad\n",b_grad)
        pad = 1
        pad_delta = np.pad(delta, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
        print(pad_delta)

        for i in range(bx): 
            # print("x_col_list\n",self.x_col_list[i])
            # print("kernel_gradient_col\n",np.dot(self.x_col_list[i].T, delta_col[i]))
            self.kernel_gradient += np.dot(self.x_col_list[i].T, delta_col[i]).T.reshape(self.kernel_gradient.shape)
        # print("kernel_gradient\n",self.kernel_gradient)

        delta_backward = np.zeros(x.shape)
        # print("kernel\n",self.kernel)
        kernel_rot = np.rot90(self.kernel, 2, (2,3))
        print("kernel_rot\n",kernel_rot)
        kernel_rot_swap = kernel_rot.swapaxes(0,1)
        print("kernel_rot_swap\n",kernel_rot_swap)
        kernel_rot_col = kernel_rot_swap.reshape(dk,-1).T
        print("kernel_col\n",kernel_rot_col)

        for i in range(bx):
            pad_delta_col = self.img2col(pad_delta[i])
            print("pad_delta_col\n",pad_delta_col)
            print("delta_backward_col\n",np.dot(pad_delta_col, kernel_rot_col))
            delta_backward[i] = np.dot(pad_delta_col, kernel_rot_col).T.reshape(dk, wx, hx)
        return delta_backward


if __name__ == "__main__":
    # img = np.arange(1,33).reshape(1,2,4,4)
    img = np.arange(1,19).reshape(1,2,3,3)
    delta = np.arange(1,13).reshape(1,3,2,2)
    print("img\n",img)
    print("delta",delta)
    t = test()
    t.forward(img)
    delta_backward = t.backward(delta,img)
    print(delta_backward)