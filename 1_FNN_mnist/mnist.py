import struct
import numpy as np
from array import array
from sklearn.model_selection import train_test_split

def load_images_labels(imagefile, labelfile):
    #load label
    labels = []
    with open(labelfile, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        label_data = array("B", file.read())
    for i in range(size):
        labels.append(label_data[i])
    #load image
    images = []
    with open(imagefile, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())
    for i in range(size):
        images.append(np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(rows * cols,1))
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
    #split validation set from train set
    train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 1/6, random_state = 0)
    #for train set, use onehot label
    train_labels_onehot = onehot_10(train_labels)
    #zip
    train_dataset = list(zip(train_images,train_labels_onehot))
    valid_dataset = list(zip(valid_images,valid_labels))
    test_dataset = list(zip(test_images,test_labels))
    return train_dataset, valid_dataset, test_dataset