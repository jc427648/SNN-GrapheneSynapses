from Plotting import plot_confusion_matrix, plotWeights, ReshapeWeights
from MNISTDataLoader import getMNIST
from STDPsynapses import STDPSynapse, LIFNeuronGroup
from Network import Network
import sklearn
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import pandas as pd
import logging
import time
import timeit
import random
import os
import optuna


def train(
    network,
    dt=0.2e-3,
    image_duration=0.05,
    n_epochs=1,
    lower_freq=20,
    upper_freq=200,
    image_threshold=50,
    n_samples=60000,
    log_interval=1000,
    det_training_accuracy=True,
    import_samples=False,
    trial=None,
):
    assert n_samples >= 0 and n_samples <= 60000, "Invalid n_samples value."
    print("Loading MNIST training samples...")
    if import_samples:
        training_data = torch.load("train_images.pt")
        training_labels = torch.load("train_labels.pt")
    else:
        training_data, training_labels = getMNIST(
            lower_freq=lower_freq,
            upper_freq=upper_freq,
            threshold=image_threshold,
            dt=dt,
            load_train_samples=True,
            load_validation_samples=False,
            load_test_samples=False,
        )[0]

    print("Training...")
    correct = 0
    start_time = timeit.default_timer()
    for epoch in range(n_epochs):
        for idx in range(n_samples):
            image, label = training_data[idx], training_labels[idx].item()
            network.OverwriteActivity()
            network.presentImage(
                image, label, image_duration, update_parameters=True)
            if det_training_accuracy and label == network.detPredictedLabel():
                correct += 1
            if (idx + 1) % log_interval == 0:
                if det_training_accuracy:
                    running_accuracy = (correct / idx) * 100
                    s_end = " - Running accuracy: %.2f%% ( %d / %d )\n" % (
                        running_accuracy,
                        correct,
                        idx + 1,
                    )
                else:
                    s_end = "\n"
                print(
                    "Training progress: sample (%d / %d) of epoch (%d / %d) - Elapsed time: %.4f%s"
                    % (
                        idx + 1,
                        n_samples,
                        epoch + 1,
                        n_epochs,
                        timeit.default_timer() - start_time,
                        s_end,
                    ),
                    end="",
                )
            network.UpdateCurrentSample()

        if trial is not None:
            trial.report((correct / idx) * 100, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if trial is not None:
        return network, (correct / idx) * 100, trial
    else:
        return network, (correct / idx) * 100


def test(
    network,
    dt=0.2e-3,
    image_duration=0.05,
    lower_freq=20,
    upper_freq=200,
    image_threshold=50,
    n_samples=10000,
    use_validation_set=False,  # Whether or not to load.use the validation set
    log_interval=1000,
    import_samples=False,


):
    assert n_samples >= 0 and n_samples <= 10000, "Invalid n_samples value."
    if use_validation_set:
        print("Loading MNIST validation samples...")
    else:
        print("Loading MNIST test samples...")
    if import_samples:
        if use_validation_set:
            test_data = torch.load("validation_images.pt")
            test_labels = torch.load("validation_labels.pt")
            print("Validating...")
        else:
            test_data = torch.load("test_images.pt")
            test_labels = torch.load("test_labels.pt")
            print("Testing...")
    else:
        MNIST_samples = getMNIST(
            lower_freq=lower_freq,
            upper_freq=upper_freq,
            threshold=image_threshold,
            dt=dt,
            load_train_samples=False,
            load_validation_samples=use_validation_set,
            load_test_samples=not use_validation_set,
            validation_samples=n_samples,
        )
        if use_validation_set:
            test_data, test_labels = MNIST_samples[1]
            print("Validating...")
        else:
            test_data, test_labels = MNIST_samples[2]
            print("Testing...")

    correct = 0
    predicted_labels = []
    start_time = timeit.default_timer()
    for idx in range(n_samples):
        image, label = test_data[idx], test_labels[idx].item()
        network.OverwriteActivity()
        network.presentImage(image, label, image_duration,
                             update_parameters=False)
        predicted_label = network.detPredictedLabel()
        predicted_labels.append(predicted_label)
        if label == predicted_label:
            correct += 1
        if (idx + 1) % log_interval == 0:
            running_accuracy = (correct / idx) * 100
            print(
                "Validation/test progress: sample (%d / %d) - Elapsed time: %.4f - Running accuracy: %.2f%% ( %d / %d )"
                % (
                    idx + 1,
                    n_samples,
                    timeit.default_timer() - start_time,
                    running_accuracy,
                    correct,
                    idx + 1,
                )
            )
        network.UpdateCurrentSample() #Placed at end of testing set

    return (correct / idx) * 100
