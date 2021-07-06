import os
import sys
from struct import unpack
import numpy as np
import sklearn
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


def unpack_MNIST_samples(
    images, labels, lower_freq, upper_freq, threshold, dt, shuffle=False
):
    images.read(4)  # Magic number
    n_images = unpack(">I", images.read(4))[0]
    rows = unpack(">I", images.read(4))[0]
    cols = unpack(">I", images.read(4))[0]
    labels.read(4)  # magic number
    n_labels = unpack(">I", labels.read(4))[0]
    assert n_images == n_labels, "Number of labels did not match number of images."
    X = torch.zeros((n_images, rows, cols), dtype=torch.uint8)  # Store all images
    y = torch.zeros((n_images, 1), dtype=torch.uint8)

    def extract_sample(i):
        if i % 20000 == 0:
            print("Progress :", i, "/", n_images)
        X[i] = torch.tensor(
            [
                [unpack(">B", images.read(1))[0] for unused_col in range(cols)]
                for unused_row in range(rows)
            ]
        )
        y[i] = unpack(">B", labels.read(1))[0]

    Parallel(require="sharedmem")(delayed(extract_sample)(i) for i in range(n_images))
    print("Progress :", n_images, "/", n_images)
    X = X.reshape([n_images, 784])
    lower_period = 1 / lower_freq
    upper_period = 1 / upper_freq
    X = torch.where(X < threshold, lower_period / dt, upper_period / dt)
    if shuffle:
        sklearn.utils.shuffle(X, y, random_state=0)

    return X, torch.squeeze(y, dim=-1)


def getMNIST(
    lower_freq=20,
    upper_freq=200,
    threshold=50,
    dt=0.2e-3,
    load_train_samples=True,  # Load training samples
    load_validation_samples=False,  # Load validation samples
    load_test_samples=False,  # Load test samples
    validation_samples=0,  # Number of samples used to construct the validation set
    export_to_disk=True,
):
    # Get MNIST samples and their corresponding labels
    if load_train_samples or load_validation_samples:
        train_images, train_labels = unpack_MNIST_samples(
            open("train-images.idx3-ubyte", "rb"),
            open("train-labels.idx1-ubyte", "rb"),
            lower_freq,
            upper_freq,
            threshold,
            dt,
            shuffle=True,
        )
        if load_validation_samples and validation_samples > 0:
            (
                train_images,
                validation_images,
                train_labels,
                validation_labels,
            ) = train_test_split(
                train_images,
                train_labels,
                test_size=validation_samples / 60000,
                random_state=1,
                shuffle=True,
                stratify=train_labels,
            )
        else:
            validation_images = None
            validation_labels = None
    else:
        validation_images = None
        validation_labels = None
        train_images = None
        train_labels = None

    if load_test_samples:
        test_images, test_labels = unpack_MNIST_samples(
            open("t10k-images.idx3-ubyte", "rb"),
            open("t10k-labels.idx1-ubyte", "rb"),
            lower_freq,
            upper_freq,
            threshold,
            dt,
            shuffle=False,
        )
    else:
        test_images = None
        test_labels = None

    if load_train_samples and export_to_disk:
        torch.save(train_images, "train_images.pt")
        torch.save(train_labels, "train_labels.pt")

    if load_validation_samples and export_to_disk:
        torch.save(validation_images, "validation_images.pt")
        torch.save(validation_labels, "validation_labels.pt")

    if load_test_samples and export_to_disk:
        torch.save(test_images, "test_images.pt")
        torch.save(test_labels, "test_labels.pt")

    return (
        (
            train_images,
            train_labels,
        ),
        (
            validation_images,
            validation_labels,
        ),
        (
            test_images,
            test_labels,
        ),
    )


if __name__ == "__main__":
    # Validate operation
    (train_data, validation_data, test_data) = getMNIST(
        load_train_samples=True,
        load_validation_samples=True,
        load_test_samples=True,
        validation_samples=10000,
        export_to_disk=True,
    )
    print(train_data[0].shape)
    print(train_data[1].shape)
    print(validation_data[0].shape)
    print(validation_data[1].shape)
    print(test_data[0].shape)
    print(test_data[1].shape)