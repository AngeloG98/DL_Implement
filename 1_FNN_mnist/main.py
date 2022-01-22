import matplotlib.pyplot as plt
import numpy as np
from mnist import mnist_process
from fnn import FNN


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = mnist_process()

    # image show
    # fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
    # ax = ax.flatten()
    # for i in range(25):
    #     ax[i].set_title(int(np.argmax(train_dataset_1[i][1])))
    #     img = train_dataset_1[i][0].reshape(28, 28)
    #     ax[i].imshow(img)
    #     # ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()

    fnn = FNN([784,30,10])
    fnn.train(train_dataset,5.0,128,5,test_dataset)
    print(fnn.predict(test_dataset))
    plt.figure("loss")
    plt.plot(fnn.loss_list)
    plt.show()

