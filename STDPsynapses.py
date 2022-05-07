import torch
import pandas as pd
import numpy as np


class STDPSynapse:
    # Defines the STDP weight and the STDP weight change. Will be used in conjunction with another object called LIF neuron.
    def __init__(self, n, wmin=-25e-3, wmax=25e-3,stdp = 0.1):
        # Initialise with random weights
        self.n = n  # Number of neurons
        # Initialise random weights, each row represents neuron, each column a different input.
        self.w = torch.zeros((n, 784)).uniform_(wmin, wmax)
        self.wmin = wmin
        self.wmax = wmax
        self.stdp = stdp

    def GetSTDP(self):
        return np.loadtxt("current.txt", delimiter=" ")
    
    def C2C_Variability(self,DelW,stdp):
        std = DelW*stdp #Convert percentage of std to raw value.
        alteredW = np.random.normal(DelW,std,1)
        return alteredW
    
    def potentiate(self, DeltaTP, Neur, STDPWindow):
        # Potentiate the synaptic weight of neuron Neur, using the DeltaT values.
        # Need to apply the lookup of the STDP window, to produce corresponding current for potentiation.
        DeltaTP = torch.round(DeltaTP * 2) / 2
        # 160 is currently hardcoded- to modularize.
        DelCurrent = STDPWindow[(DeltaTP[0 : len(DeltaTP)] * 1 + 85).long()]#Refer to previous code for why this is.
        deltaW = torch.multiply(Neur, DelCurrent)
        deltaW = self.C2C_Variability(deltaW,self.stdp) #10% stdp
        self.w += deltaW
        # Bound the weights
        self.w = torch.clamp(self.w, self.wmin, self.wmax)

    def depress(self, DeltaTN, Neur, STDPWindow):
        # Depress the value synaptic weight of neuron Neur, using the values of DeltaT
        # This rounding allows simple implementation of this specific STDP window.
        DeltaTN = torch.round(DeltaTN * 2) / 2
        # 160 is currently hardcoded- to modularize.
        DelCurrent = STDPWindow[(DeltaTN[0 : len(DeltaTN)] * 1 + 85).long()]
        deltaW = torch.multiply(Neur, DelCurrent)
        deltaW = self.C2C_Variability(deltaW,self.stdp) #10% stdp
        self.w += deltaW
        # Bound the weights
        self.w = torch.clamp(self.w, self.wmin, self.wmax)




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
        self.Vth = 5 * torch.ones_like(torch.Tensor(n))
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
