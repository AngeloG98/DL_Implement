import matplotlib.pyplot as plt
import numpy as np
from mnist_conv import mnist_process_conv

if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = mnist_process_conv()

    # image show
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(25):
        ax[i].set_title(int(np.argmax(train_dataset[i][1])))
        img = train_dataset[i][0]
        # ax[i].imshow(img)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()