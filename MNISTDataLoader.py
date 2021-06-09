import os
import sys
from struct import unpack
import numpy as np


def getMNIST(mode='train', lower_freq=20, upper_freq=200, threshold=50, dt=0.2e-3):
    # Get the MNIST images and their corresponding labels as a series of tuples.STDPsynapse N is number of images.
    assert mode in ['train', 'test'], 'Invalid mode specified.'
    if (mode == 'train'):
        images = open('train-images.idx3-ubyte', 'rb')
        labels = open('train-labels.idx1-ubyte', 'rb')
    else:
        images = open('t10k-images.idx3-ubyte', 'rb')
        labels = open('t10k-labels.idx1-ubyte', 'rb')

    images.read(4)  # Magic number
    n_images = unpack('>I', images.read(4))[0]
    rows = unpack('>I', images.read(4))[0]
    cols = unpack('>I', images.read(4))[0]
    labels.read(4)  # magic number
    n_labels = unpack('>I', labels.read(4))[0]

    assert n_images == n_labels, 'Number of labels did not match number of images.'
    X = np.zeros((n_images, rows, cols), dtype=np.uint8)  # Store all images
    y = np.zeros((n_images, 1), dtype=np.uint8)

    for i in range(n_images):
        if (i % 1000 == 0):
            print('Progress :', i, '/', n_images)
        X[i] = [[unpack('>B', images.read(1))[0] for unused_col in
                 range(cols)] for unused_row in range(rows)]
        y[i] = unpack('>B', labels.read(1))[0]

    print('Progress :', n_images, '/', n_images, '\n')

    # These values are in between 0 and 255, but in should be 20 and 200.Need to fix
    X = X.reshape([n_images, 784])
    lower_period = 1 / lower_freq
    upper_period = 1 / upper_freq

    # It's also been found that spike times might the use index and not actual value. Need to reflect in choice of frequencies.
    # Converting to binary image
    X = np.where(X < threshold, lower_period/dt, upper_period/dt)
    return {'X': X, 'y': y}


if __name__ == "__main__":
    # Validate operation
    training_data = getMNIST()
    print(training_data['X'].shape)
    print(training_data['y'].shape)
    validation_data = getMNIST(mode='test')
    print(validation_data['X'].shape)
    print(validation_data['y'].shape)
