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


def train(
    network,
    n_output_neurons=30,
    dt=0.2e-3,
    image_duration=0.05,
    n_epochs=1,
    lower_freq=20,
    upper_freq=200,
    image_threshold=50,
    n_samples=60000,
    log_interval=1000,
    det_training_accuracy=True,
):
    assert n_samples >= 0 and n_samples <= 60000, "Invalid n_samples value."
    print("Loading MNIST training samples...")
    training_data, training_labels = getMNIST(
        mode="train",
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        threshold=image_threshold,
        dt=dt,
    )
    print("Training...")
    correct = 0
    start_time = timeit.default_timer()  # Start timer
    for epoch in range(n_epochs):
        for idx in range(n_samples):
            image, label = training_data[idx], training_labels[idx].item()
            network.OverwriteActivity()
            network.presentImage(image, label, image_duration, update_parameters=True)
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
                # plotWeights(
                #     ReshapeWeights(network.synapse.w, n_output_neurons)[0],
                #     network.synapse.wmax,
                #     network.synapse.wmin,
                #     title="idx_%d" % (idx + 1),
                # )
                network.save()
            network.UpdateCurrentSample()
    return network, (correct / idx) * 100


def test(
    network,
    n_output_neurons=30,
    dt=0.2e-3,
    image_duration=0.05,
    lower_freq=20,
    upper_freq=200,
    image_threshold=50,
    n_samples=10000,
    log_interval=1000,
):
    assert n_samples >= 0 and n_samples <= 10000, "Invalid n_samples value."
    print("Loading MNIST validation/test samples...")
    test_data, test_labels = getMNIST(
        mode="test",
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        threshold=image_threshold,
        dt=dt,
    )
    correct = 0
    predicted_labels = []
    print("Validating/Testing...")
    start_time = timeit.default_timer()  # Start timer
    for idx in range(n_samples):
        image, label = test_data[idx], test_labels[idx].item()
        network.OverwriteActivity()
        network.presentImage(image, label, image_duration, update_parameters=False)
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
    cf = confusion_matrix(
        test_labels.numpy()[0:n_samples], predicted_labels, normalize="true"
    )
    plot_confusion_matrix(cf)
    return (correct / idx) * 100


def main(
    n_output_neurons=30,  # Number of output neurons
    n_samples_memory=30,  # Number of samples to retain in memory (to determine class allocations)
    dt=0.2e-3,  # Timestep (s)
    image_duration=0.05,  # Duration (s) to present each image for
    n_epochs=1,  # Number of training epochs
    lower_freq=20,  # Lower encoding frequency
    upper_freq=200,  # Upper encoding frequency
    image_threshold=50,  # Threshold to generate greyscale input images
    n_samples_train=60000,  # Number of training samples per epoch
    n_samples_test=10000,  # Number of validation/test samples
    Ve=0.0,
    tau=0.1,
    R=1000,
    gamma=0.005,
    target_activity=10,
    v_th_min=0.25,
    v_th_max=50,
    fixed_inhibition_current=-1.0,
    log_interval=1000,  # log interval for train() and test() methods
    det_training_accuracy=True,  # Boolean to determine whether or not if the training accuracy is determined
    output_dir="output",
):
    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_samples_memory,
        Ve=Ve,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=target_activity,
        v_th_min=v_th_min,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
        output_dir=output_dir,
    )  # Define the network architecture
    network, training_set_accuracy = train(
        network,
        n_samples=n_samples_train,
        dt=dt,
        image_duration=image_duration,
        n_epochs=n_epochs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        log_interval=log_interval,
        det_training_accuracy=det_training_accuracy,
    )
    # Test the network
    # network.load() This method can be used to load network parameters exported using .save()
    # test_set_accuracy = test(
    #     network,
    #     n_samples=n_samples_test,
    #     dt=dt,
    #     image_duration=image_duration,
    #     lower_freq=lower_freq,
    #     upper_freq=upper_freq,
    #     image_threshold=image_threshold,
    #     log_interval=log_interval,
    # )  # Validate/Test the Network
    return training_set_accuracy


if __name__ == "__main__":
    main(
        n_samples_train=10000,
        log_interval=1000,
        det_training_accuracy=True,
        target_activity=30,
    )
