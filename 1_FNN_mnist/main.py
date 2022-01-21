import matplotlib.pyplot as plt
from mnist import mnist_process
from fnn import FNN


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = mnist_process()
    fnn = FNN([784,30,10])
    fnn.train(train_dataset,5.0,128,5,test_dataset)
    print(fnn.predict(test_dataset))
    plt.figure("loss")
    plt.plot(fnn.loss_list)
    plt.show()