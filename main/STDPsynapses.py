import torch
import pandas as pd
import numpy as np
# from plotting import plotWeights
# Need some sort of import for the STDP window


class STDPSynapse:
    # defines the STDP weight and the STDP weight change. Will be used in conjunction with another object called LIF neuron.
    def __init__(self, n, wmin=-25e-3, wmax=25e-3):
        # could potentially use a dictionary for the STDP window, or just what you created in MATLAB.

        # Initialise with random weights
        self.n = n  # Number of neurons
        # Initialise random weights, each row represents neuron, each column a different input.
        self.w = 50e-3 * torch.rand(n, 784) - 25e-3
        self.wmin = wmin
        self.wmax = wmax

    def potentiate(self, DeltaTP, Neur, STDPWindow):
        # Potentiate the synaptic weight of neuron Neur, using the DeltaT values.
       # Need to apply the lookup of the STDP window, to produce corresponding current for potentiation.
        DeltaTP = torch.round(DeltaTP * 2) / 2
        DelCurrent = torch.zeros(len(DeltaTP))

        for i in range(len(DeltaTP)):
            # Should convert whole tensor to float prior to this for speed increase
            DelCurrent[i] = STDPWindow[float(DeltaTP[i])]

        # Need to be careful with torch and numpy, it could create some errors.
        deltaW = torch.multiply(Neur, DelCurrent)
        self.w += deltaW

        # Make sure weights are within the bounds
        self.w = torch.clamp(self.w, self.wmin, self.wmax)

    def depress(self, DeltaTN, Neur, STDPWindow):
        # Depress the value synaptic weight of neuron Neur, using the values of DeltaT
        # This rounding allows simple implemenation of this specific STDP window.
        DeltaTN = torch.round(DeltaTN * 2) / 2

        DelCurrent = torch.zeros(len(DeltaTN))

        for i in range(len(DeltaTN)):
            DelCurrent[i] = STDPWindow[float(DeltaTN[i])]

        deltaW = torch.multiply(Neur, DelCurrent)
        self.w += deltaW

        # Bound the weights
        self.w = torch.clamp(self.w, self.wmin, self.wmax)

    def GetSTDP(self):
        # Use the following lines to get the dictionary for the STDP window.

        b = np.loadtxt('STDPWindow.txt', delimiter=",")
        d = {}
        for i in range(len(b[0, :])):
            # Probably don't need conversion, we'll see*1e-3 #Convert to seconds
            key = b[0, i]
            val = b[1, i]
            d[key] = val
# Probably makes more sense to have STDP window stored at synapse level
        return d


class LIFNeuronGroup:
    # Defines the neuron parameters used to perform the STDP classification.
    def __init__(self, n, mode, Ve=0.2, tau=0.6, R=100, gamma=0.01, target=35, VthMin=0.02, VthMax=20):
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
        self.Vth = 5 * torch.ones_like(torch.Tensor(n))
        # Determine the occurrances of post-synaptic spikes.
        self.s = torch.zeros_like(torch.Tensor(n))

    def step(self, dt, current, sumAct, mode):
        # Note: current is the net current presented to the network.
        self.v += dt / self.tau * (self.Ve - self.v + self.R * current)

        self.s = (self.v >= self.Vth)  # Check for spiking events.
        self.v[self.s] = self.Ve
        self.v[self.v < self.Ve] = self.Ve

        # I think activity should be monitored at the network level, but the threshold should still be updated.
        # Update adaptive threshold
        if (mode == 'train'):
            self.Vth += dt * self.gamma * (sumAct - self.target)
            self.Vth = torch.clamp(self.Vth, self.VthMin, self.VthMax)
