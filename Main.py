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
    lower_freq=200,
    upper_freq=5,
    image_threshold=200,
    n_samples=60000,
    log_interval=5000,
    det_training_accuracy=True,
    data=None,
    labels=None,
    trial=None,
):
    assert n_samples >= 0 and n_samples <= 60000, "Invalid n_samples value."
    assert data is not None
    assert labels is not None
    print("Training...")
    correct = 0
    start_time = timeit.default_timer()
    for epoch in range(n_epochs):
        for idx in range(n_samples):
            image, label = data[idx], labels[idx].item()
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

    return network, (correct / idx) * 100


def test(
    network,
    dt=0.2e-3,
    image_duration=0.05,
    lower_freq=20,
    upper_freq=200,
    image_threshold=50,
    n_samples=10000,
    log_interval=5000,
    data=None,
    labels=None
):
    assert n_samples >= 0 and n_samples <= 10000, "Invalid n_samples value."
    assert data is not None
    assert labels is not None
    correct = 0
    predicted_labels = []
    start_time = timeit.default_timer()
    for idx in range(n_samples):
        image, label = data[idx], labels[idx].item()
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
        network.UpdateCurrentSample()

    return (correct / idx) * 100