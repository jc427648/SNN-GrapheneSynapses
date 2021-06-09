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
dt = 0.2e-3  # Timestep (s)
image_duration = 0.05  # Duration (s) to present each image for
n_epochs = 1  # Number of training epochs
lower_freq = 20  # Lower encoding frequency
upper_freq = 200  # Upper encoding frequency
image_threshold = 50  # Threshold to generate greyscale input images


# confusion = torch.zeros((10, 10))
# accuracy = torch.zeros(10)
# TotalResp = torch.zeros(n)  # Don't forget to hardcode these values
# OverAccuracy = 0
# ConSum = 0
# correct = 0
# NoResp = 0

# PatCount = 0  # Initialise PatCount

# if (mode == 'test'):
#     network.synapse.w = torch.load('WeightMatrix.pt')
#     network.group.Vth = torch.load('FinalThresholds.pt')
#     classes = torch.reshape(torch.load('NeuronAssignments.pt'), (n, 1))

# if mode == 'test':
#     TotalResp += network.Activity[:, PatCount]
#     values, indices = torch.max(
#         network.Activity[:, PatCount], 0, keepdim=True)
#     # Shouldn't be indices, should be neuron associated with index
#     if label[0] == classes[indices]:
#         correct += 1
#     if torch.sum(network.Activity[:, PatCount], 0, keepdim=True) == 0:
#         NoResp += 1
# # elif prog%1 == 0:
#     # Need to save the re-arranged weights as some sort of image. Also need to somehow need to run testing as well.
#  #   fileString = '%dImages.pt' %(prog)
#   #  weights, assignments = ReshapeWeights(network.synapse.w,n)
#    # torch.save(weights,os.path.join(EvoPath,fileString))

# PatCount += 1


# if mode == 'train':
#     torch.save(network.synapse.w, 'WeightMatrix.pt')
#     torch.save(network.group.Vth, 'FinalThresholds.pt')
#     # Rearrange the weights for plotting

#     values, assignments = network.Assignment.max(axis=1)
#     RWeights, assignments = ReshapeWeights(network.synapse.w, n, assignments)

#     torch.save(assignments, 'NeuronAssignments.pt')
#     plotWeights(RWeights, assignments,
#                 network.synapse.wmax, network.synapse.wmin)
#     print(network.group.Vth)
#     output, counts = torch.unique(assignments, return_counts=True)
#     print(counts)
# elif mode == 'test':

#     for i in range(n):
#         confusion[int(classes[i]), :] += network.Assignment[i, :]
#     # assignments is wrong, as assignments is only from 1 to 10
#     RWeights = torch.zeros(280, 280)
#     ConSum = torch.sum(confusion)
#     for i in range(10):
#         accuracy[i] = confusion[i, i]/torch.sum(confusion[i, :])
#         OverAccuracy += confusion[i, i]
#     numResponse = confusion
#     confusion = confusion/torch.max(confusion)  # normalise confusion matrix
#     OverAccuracy = OverAccuracy/ConSum  # Overall accuracy
#     # Think about saving the confusion matrix as well just in case.

#     RWeights, assignments = ReshapeWeights(network.synapse.w, n)

#     plotConfusion(RWeights, confusion, network.synapse.wmax,
#                   network.synapse.wmin)
#     print(accuracy)
#     print('\n')
#     print(OverAccuracy)
#     print('\n')
#     print(numResponse)
#     # It would be better to have a plot accuracy function.
#     Acc = correct/(NImages*repetitions)
#     print('\n')
#     print(Acc)
#     print(NoResp)
#     print('\n')
#     print(TotalResp)


if __name__ == "__main__":
    print('Initializing...')
    # Define the network architecture
    network = Network(n_output_neurons, dt=dt)
    # Train the network
    training_data, training_labels = getMNIST(
        lower_freq=lower_freq, upper_freq=upper_freq, threshold=image_threshold, dt=dt)
    start_time = timeit.default_timer()  # Start timer
    for epoch in range(n_epochs):
        # print(training_labels.shape)
        # exit(0)
        n_samples = training_labels.size
        for idx in range(n_samples):
            image, label = training_data[idx], training_labels[idx]
            network.presentImage(image, label, image_duration)
            if (idx % 1000 == 0):
                print('Training progress: sample (%d / %d) of epoch (%d / %d) - Elapsed time: %.4f'
                      % (idx, n_samples, epoch, n_epochs, timeit.default_timer() - start_time))

    values, assignments = network.Assignment.max(axis=1)
    r_weights, assignments = ReshapeWeights(
        network.synapse.w, n_output_neurons, assignments)
    plotWeights(r_weights, assignments,
                network.synapse.wmax, network.synapse.wmin)
    #         # Validate (test) the network
    # test_data, test_labels = getMNIST(mode='test',
    #                                   lower_freq=lower_freq,
    #                                   upper_freq=upper_freq,
    #                                   threshold=image_threshold,
    #                                   dt=dt,
    #                                   shuffle=False)
