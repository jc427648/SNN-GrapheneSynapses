import torch
import pandas as pd
import numpy as np


class STDPSynapse:
    # Defines the STDP weight and the STDP weight change. Will be used in conjunction with another object called LIF neuron.
    def __init__(self, n, wmin=-10e-6, wmax=10e-6):
        # Initialise with random weights
        self.n = n  # Number of neurons
        # Initialise random weights, each row represents neuron, each column a different input.
        self.w = torch.zeros((n, 784)).uniform_(wmin, wmax)
        self.wmin = wmin
        self.wmax = wmax

    def GetSTDP(self):
        return np.loadtxt("current.txt", delimiter=" ")


class LIFNeuronGroup:
    # Defines the neuron parameters used to perform the STDP classification.
    def __init__(
        self, n, Ve=0.2, tau=0.6, R=100, gamma=0.01, target=35, VthMin=0.02, VthMax=20
    ):
        # Is this init affecting my other inits? I don't think, so modification happens anyway.
        self.n = n
        self.Ve = Ve  # Resting potential
        self.tau = tau  # Timing constant
        self.R = R  # Membrane resistance
        self.gamma = gamma  # homeostasis constant
        self.target = target  # Target activity for all neurons.
        self.VthMin = VthMin  # minimum threshold
        self.VthMax = VthMax  # Maximum threshold.
        # Membrane potential at current point in time.
        self.v = self.Ve * torch.ones_like(torch.Tensor(n))
        # Randomise intial thresholds
        # self.Vth = (VthMax-VthMin)*torch.rand(n)+torch.ones_like(torch.Tensor(n))*VthMin#The thresholds for each neuron, initially Vthmin.
        self.Vth = VthMin * torch.ones_like(torch.Tensor(n))
        # Determine the occurrances of post-synaptic spikes.
        self.s = torch.zeros_like(torch.Tensor(n))

    def step(self, dt, current, sumAct, update_parameters=True):
        # Note: current is the net current presented to the network.
        self.v += dt / self.tau * (self.Ve - self.v + self.R * current)
        self.s = self.v >= self.Vth  # Check for spiking events.
        self.v[self.s] = self.Ve
        self.v[self.v < self.Ve] = self.Ve
        # I think activity should be monitored at the network level, but the threshold should still be updated.
        # Update adaptive threshold
        if update_parameters:
            self.Vth += dt * self.gamma * (sumAct - self.target)
            self.Vth = torch.clamp(self.Vth, self.VthMin, self.VthMax)
