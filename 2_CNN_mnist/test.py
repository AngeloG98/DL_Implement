import numpy as np

def img2col(xi,kernel):
    wx, hx, dx = xi.shape
    wk, hk, dk, nk = kernel.shape
    wfeature = (wx - wk) + 1
    hfeature = (hx - hk) + 1
    image_col = np.zeros((wfeature*hfeature, wk*hk*dk))
    idx = 0
    for i in range(wfeature):
        for j in range(hfeature):
            image_col[idx] = xi[i:i + wk, j:j + hk, :].reshape(-1)
            idx += 1
    return image_col

def img2col2(xi,kernel):
    wx, hx, dx = xi.shape
    wk, hk, dk, nk = kernel.shape
    wfeature = (wx - wk) + 1
    hfeature = (hx - hk) + 1
    image_col = np.zeros((wfeature*hfeature, wk*hk*dk))
    for deep in range(dx):
        temp = np.zeros((wfeature*hfeature, wk*hk))
        idx = 0
        for i in range(wfeature):
            for j in range(hfeature):
                temp[idx] = xi[i:i + wk, j:j + hk, deep].reshape(-1)
                idx += 1
        image_col[:,deep*wk*hk:(deep+1)*wk*hk] =temp
    return image_col

def flip(arr):
    arr = np.flipud(arr)
    arr = np.fliplr(arr)
    return arr



def matrix_rotation():
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    data1 = flip(data)
    print("data: ", data)
    print("data1: ", data1)
def flip180():
    data = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    data180 = np.rot90()
if __name__ == "__main__":
    kernel = np.random.random((2, 2, 2, 2))
    # print("0:",kernel)
    kernel_rot = np.rot90(kernel, 2)
    # print("1:",kernel_rot)
    kernel_rot = kernel_rot.swapaxes(2,3)
    # print("2:",kernel_rot)
    kernel_rot1 = kernel[::-1,::-1]
    # print("3:",kernel_rot1)
    kernel_rot_col = kernel_rot.reshape(-1, 2)
    kernel_rot1_col = kernel_rot1.reshape(-1, 2)
    x = np.random.random((3, 3, 2))
    print("x",x)
    x_col = img2col(x,kernel)
    print("xx",x_col)
    x_col2 = img2col2(x,kernel)
    print("xx2",x_col2)
    print()