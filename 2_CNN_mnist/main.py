import matplotlib.pyplot as plt
import numpy as np
from mnist_conv import mnist_process_conv
from cnn import CNN
if __name__ == "__main__":
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = mnist_process_conv()

    # image show
    # fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
    # ax = ax.flatten()
    # for i in range(25):
    #     ax[i].set_title(int(np.argmax(train_Y[i])))
    #     img = train_X[i]
    #     # ax[i].imshow(img)
    #     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()

    cnn = CNN()
    # cnn.train(train_X, train_Y, valid_X, valid_Y)
    cnn.eval(test_X, test_Y, pre_train = True, filename = "2_CNN_mnist/model/mnist_cnn_model_0.npz")
