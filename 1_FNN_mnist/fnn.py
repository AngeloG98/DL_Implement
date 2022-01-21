import struct
import numpy as np
from array import array
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_images_labels(imagefile, labelfile):
    labels = []
    with open(labelfile, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        label_data = array("B", file.read())
    for i in range(size):
        labels.append(label_data[i])
    images = []
    with open(imagefile, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())
    for i in range(size):
        images.append(np.array(image_data[i * rows * cols:(i + 1) * rows * cols]))

    return images, labels

def onehot_10(labels):
    labels_onehot = []
    for lab in labels:
        lab_onehot = np.zeros((10,1))
        lab_onehot[lab] = 1.0
        labels_onehot.append(lab_onehot)
    return np.array(labels_onehot)


def mnist_process():
    train_imagefile = '0_dataset/MNIST/train-images.idx3-ubyte'
    train_labelfile = '0_dataset/MNIST/train-labels.idx1-ubyte'
    test_imagefile = '0_dataset/MNIST/t10k-images.idx3-ubyte'
    test_labelfile = '0_dataset/MNIST/t10k-labels.idx1-ubyte'
    images, labels = load_images_labels(train_imagefile, train_labelfile)
    test_images, test_labels = load_images_labels(test_imagefile, test_labelfile)
    train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 1/6, random_state = 0)
    train_labels_onehot = onehot_10(train_labels)
    train_dataset = list(zip(train_images,train_labels_onehot))
    print()

if __name__ == "__main__":
    mnist_process()
