from MNISTDataLoader import getMNIST
from STDPsynapses import STDPSynapse, LIFNeuronGroup
from Network import Network
from plotting import plotWeights, plotConfusion, ReshapeWeights

import torch
import numpy as np
import pandas as pd
import logging
import time
import timeit
import random
import os

# Parameters and constants
n_output_neurons = 30  # Number of output neurons
# Number of samples to retain in memory (to determine class allocations)
n_samples_memory = 30
dt = 0.2e-3  # Timestep (s)
image_duration = 0.05  # Duration (s) to present each image for
n_epochs = 1  # Number of training epochs
lower_freq = 20  # Lower encoding frequency
upper_freq = 200  # Upper encoding frequency
image_threshold = 50  # Threshold to generate greyscale input images

if __name__ == "__main__":
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Loading MNIST training samples...')
    # Define the network architecture
    network = Network(n_output_neurons, n_samples_memory, dt=dt)
    # Train the network
    training_data, training_labels = getMNIST(
        lower_freq=lower_freq, upper_freq=upper_freq, threshold=image_threshold, dt=dt)
    start_time = timeit.default_timer()  # Start timer
    for epoch in range(n_epochs):
        n_samples = training_labels.numel()
        for idx in range(n_samples):
            image, label = training_data[idx], training_labels[idx]
            network.presentImage(image, label, image_duration)
            if (idx % 1000 == 0):
                print('Training progress: sample (%d / %d) of epoch (%d / %d) - Elapsed time: %.4f'
                      % (idx, n_samples, epoch + 1, n_epochs, timeit.default_timer() - start_time))
                plotWeights(ReshapeWeights(network.synapse.w, n_output_neurons)[0],
                            network.synapse.wmax, network.synapse.wmin, title='idx_%d' % idx)
                if idx == 1000:
                    break
