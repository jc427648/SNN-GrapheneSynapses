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


confusion = torch.zeros((10, 10))
accuracy = torch.zeros(10)
TotalResp = torch.zeros(n)  # Don't forget to hardcode these values
OverAccuracy = 0
ConSum = 0
correct = 0
NoResp = 0

if mode == 'train':
    NImages = 60000
elif mode == 'test':
    NImages = 10000  # Can go up to 60000, but paper uses 40000

repetitions = 1

# time = 0.05  # Time in seconds of image presentation

PatCount = 0  # Initialise PatCount

if (mode == 'test'):
    network.synapse.w = torch.load('WeightMatrix.pt')
    network.group.Vth = torch.load('FinalThresholds.pt')
    classes = torch.reshape(torch.load('NeuronAssignments.pt'), (n, 1))

data = network.GetMNIST(NImages, mode)  # Images and labels

X, y = data['X'], data['y']

# Define the pathname for weight evolution folder to store the weights in the folder
EvoPath = os.path.join(os.getcwd(), 'Weight evolution')


start = timeit.default_timer()  # Start timer

# Loop through all images several times
for reps in range(repetitions):
    # Loop through every image randomly
    r = list(range(NImages))
    random.shuffle(r)
    prog = 0
    for idx in r:
        # Generate the spike trains for the specific image
        image, label = X[idx], y[idx]
        spikes, spike_times = network.GenSpkTrain(image, time)

        if PatCount == n-1:  # Need to update this so that the variable makes sense
            PatCount = 0
        # Set the activity so that past value is ignored
        network.resetActivity(PatCount)
        # RESET VOLTAGES (NEW cODE!!!) May also want to reset current
        network.group.v[:] = network.group.Ve
        network.current[:] = 0
        network.CurrCtr[:] = 0
        network.InhibVec[:] = 0
        network.InhibCtr[:] = 0
        # Run the image
        network.run(mode, spikes, spike_times, time, PatCount)
        prog += 1
        # Print the progress
        print('Training progress: (%d / %d) - Elapsed time: %.4f' %
              (prog, NImages, timeit.default_timer() - start))
        # Add to label number
        network.setAssignment(label[0], PatCount)

        # Should check and save every 1000 images, preferably saving the rearranged weights. You can use the % operator
        # in python to format strings like sprintf. You will need to develop some other code to be used

        # Should have code here to determine for testing phase to determine if correct or not.
        if mode == 'test':
            TotalResp += network.Activity[:, PatCount]
            values, indices = torch.max(
                network.Activity[:, PatCount], 0, keepdim=True)
            # Shouldn't be indices, should be neuron associated with index
            if label[0] == classes[indices]:
                correct += 1
            if torch.sum(network.Activity[:, PatCount], 0, keepdim=True) == 0:
                NoResp += 1
        # elif prog%1 == 0:
            # Need to save the re-arranged weights as some sort of image. Also need to somehow need to run testing as well.
         #   fileString = '%dImages.pt' %(prog)
          #  weights, assignments = ReshapeWeights(network.synapse.w,n)
           # torch.save(weights,os.path.join(EvoPath,fileString))

        PatCount += 1

if mode == 'train':
    torch.save(network.synapse.w, 'WeightMatrix.pt')
    torch.save(network.group.Vth, 'FinalThresholds.pt')
    # Rearrange the weights for plotting

    values, assignments = network.Assignment.max(axis=1)
    RWeights, assignments = ReshapeWeights(network.synapse.w, n, assignments)

    torch.save(assignments, 'NeuronAssignments.pt')
    plotWeights(RWeights, assignments,
                network.synapse.wmax, network.synapse.wmin)
    print(network.group.Vth)
    output, counts = torch.unique(assignments, return_counts=True)
    print(counts)
elif mode == 'test':

    for i in range(n):
        confusion[int(classes[i]), :] += network.Assignment[i, :]
    # assignments is wrong, as assignments is only from 1 to 10
    RWeights = torch.zeros(280, 280)
    ConSum = torch.sum(confusion)
    for i in range(10):
        accuracy[i] = confusion[i, i]/torch.sum(confusion[i, :])
        OverAccuracy += confusion[i, i]
    numResponse = confusion
    confusion = confusion/torch.max(confusion)  # normalise confusion matrix
    OverAccuracy = OverAccuracy/ConSum  # Overall accuracy
    # Think about saving the confusion matrix as well just in case.

    RWeights, assignments = ReshapeWeights(network.synapse.w, n)

    plotConfusion(RWeights, confusion, network.synapse.wmax,
                  network.synapse.wmin)
    print(accuracy)
    print('\n')
    print(OverAccuracy)
    print('\n')
    print(numResponse)
    # It would be better to have a plot accuracy function.
    Acc = correct/(NImages*repetitions)
    print('\n')
    print(Acc)
    print(NoResp)
    print('\n')
    print(TotalResp)


if __name__ == "__main__":
    # Define the network architecture
    network = Network(n_output_neurons, dt=dt)
    # Train the network
    for i in range(n_epochs):
        pass
