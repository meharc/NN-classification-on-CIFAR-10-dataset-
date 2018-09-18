# visualizing NN classification using PCA
# PS : change the location of the train and test files

import numpy as np
from sklearn.decomposition import PCA
import pickle
import ASS1.part1
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():

    # getting the data
    batch = []
    batch.append(unpickle(
        'C:\\Users\MEHAR CHATURVEDI\PycharmProjects\DLFCV\ASS1\cifar-10-batches-py\data_batch_1'))
    batch.append(unpickle(
        'C:\\Users\MEHAR CHATURVEDI\PycharmProjects\DLFCV\ASS1\cifar-10-batches-py\data_batch_2'))
    batch.append(unpickle(
        'C:\\Users\MEHAR CHATURVEDI\PycharmProjects\DLFCV\ASS1\cifar-10-batches-py\data_batch_3'))
    batch.append(unpickle(
        'C:\\Users\MEHAR CHATURVEDI\PycharmProjects\DLFCV\ASS1\cifar-10-batches-py\data_batch_4'))
    batch.append(unpickle(
        'C:\\Users\MEHAR CHATURVEDI\PycharmProjects\DLFCV\ASS1\cifar-10-batches-py\data_batch_5'))

    train_features, train_labels = ASS1.part1.get_train_features(batch)

    test_batch = unpickle(
        'C:\\Users\MEHAR CHATURVEDI\PycharmProjects\DLFCV\ASS1\cifar-10-batches-py\\test_batch')

    test_features, test_labels = ASS1.part1.get_test_features(test_batch)

    pca = PCA(n_components=2)
    X = pca.fit_transform(train_features)
    plt.scatter(X[:, 0], X[:, 1], c=train_labels, marker='.', cmap='tab10')
    plt.colorbar()
    X_temp=pca.transform(test_features)
    plt.scatter(X_temp[:,0],X_temp[:,1],c=test_labels,marker='-',cmap='tab10')
    plt.show()


if __name__ == '__main__':
    main()
