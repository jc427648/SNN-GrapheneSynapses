import torch
import os
import sys
from struct import unpack
import numpy as np
from STDPsynapses import STDPSynapse, LIFNeuronGroup
import copy  # TEMP


class Network():
    def __init__(self, n_output_neurons=30, n_samples_memory=30, dt=0.2e-3):
        self.synapse = STDPSynapse(n_output_neurons, wmin=-45e-3, wmax=45e-3)
        self.group = LIFNeuronGroup(
            n_output_neurons, Ve=0.0, tau=0.1, R=1000, gamma=0.005, target=10, VthMin=0.25, VthMax=50)
        self.dt = dt
        self.n_output_neurons = n_output_neurons
        self.n_samples_memory = n_samples_memory
        self.current_sample = 0
        self.Activity = torch.zeros(n_output_neurons, n_samples_memory)
        self.sumAct = torch.zeros(n_output_neurons)
        self.STDPWindow = self.synapse.GetSTDP()
        self.Assignment = torch.zeros((n_output_neurons, 10))
        # Current Counter (not correct, should be a vector)
        self.CurrCtr = torch.zeros((784))
        # Inhibition counter (May not be needed)
        self.InhibCtr = torch.zeros((self.n_output_neurons))
        self.InhibVec = torch.zeros((self.n_output_neurons))
        self.current = torch.zeros((self.n_output_neurons))

    def run(self, spikes, spike_times, time, update_parameters=True):
        # in this instance, mode is either train or test, and time is the time the image is presented.
        # My thinking is this, have another method dedicated to getting the MNIST and generating the spike train.
        # Then do the steps for all fo the time range, make sure to log the info to get idea of time.

        # Current should be a nx1 vector of the sum of all currents (including inhibition.)
        # Consider having creating one more file to run the entire MNIST simulation.
        c_p_w = 1e-3  # Current Pulse Width (s)
        i_p_w = 1e-3  # Inhibition Pulse Width (s)
        inhib = -1.0  # Fixed inhibition current (A)

        for t in range(int(time/self.dt)):
            # Work out the current, then step the voltages, and then decrement current counters.
            # Spikes is a tensor, not a list.
            self.CurrCtr[spikes[t, :] == 1] = c_p_w
            # Apply inhibition.

            # print(self.synapse.w.shape)
            # exit(0)

            currentMat = torch.multiply(
                self.CurrCtr > 0, self.synapse.w)  # Matrix of all values
            self.current = torch.sum(currentMat, 1)
            self.current += self.InhibVec

            # Decrement the current pulse widths.
            self.group.step(self.dt, self.current,
                            self.sumAct, update_parameters)
            self.CurrCtr -= self.dt
            self.InhibCtr -= self.dt

            if torch.sum(self.group.s) > 0:
                if update_parameters:
                    # Update synaptic weights
                    DeltaT = t-spike_times
                    # Ensure time is 80ms long (off STDP)
                    DeltaT[DeltaT == t] = 80e-3/self.dt

                    # print(DeltaT)
                    # exit(0)

                    # DeltaTP = copy.deepcopy(DeltaT)
                    # DeltaTP[DeltaT <= 0] = 400
                    try:
                        DeltaTP = DeltaT[DeltaT > 0].min(axis=0)[0]
                    except:
                        DeltaTP = torch.Tensor([400])
                    # print(DeltaTP.ndim)
                    # print(len(DeltaTP))
                    if DeltaTP.ndim == 0:
                        DeltaTP = torch.Tensor([400])

                    # DeltaTP, indices = torch.where(DeltaT > 0, DeltaT, 400*torch.ones(
                    # #     784, dtype=torch.int64)).min(axis=0)  # 400 should be some variable
                    # print(DeltaTP)
                    # # print(DeltaTP2)
                    # exit(0)

                    DeltaTP = 1e3*self.dt*DeltaTP

                    # DeltaTN = copy.deepcopy(DeltaT)
                    # DeltaTN[DeltaT >= 0] = 400
                    try:
                        DeltaTN = DeltaT[DeltaT < 0].max(axis=0)[0]
                    except:
                        DeltaTN = torch.Tensor([-400])
                    # print(len(DeltaTN))
                    if DeltaTN.ndim == 0:
                        DeltaTN = torch.Tensor([-400])
                    # DeltaTN, indices = torch.where(
                    #     DeltaT < 0, DeltaT, -400*torch.ones(784, dtype=torch.int64)).max(axis=0)
                    DeltaTN = 1e3*self.dt*DeltaTN

                    Neur = torch.unsqueeze(self.group.s, 1)

                    self.synapse.potentiate(DeltaTP, Neur, self.STDPWindow)
                    self.synapse.depress(DeltaTN, Neur, self.STDPWindow)

                # Update activity

                # print(self.group.s.shape)
                # print(self.Activity.shape)
                # exit(0)

                # self.Activity[:, self.current_sample] += self.group.s
                self.sumAct += self.group.s

                # Update inhibition
                self.InhibCtr[torch.logical_not(self.group.s)] = i_p_w
                self.InhibVec = torch.multiply(self.InhibCtr > 0, inhib)

        self.current_sample += 1
        if self.current_sample == self.n_samples_memory:
            self.current_sample = 0

    def GenSpkTrain(self, image, time):
        # Generate Poissonian spike times to input into the network. Note time is integer.
        # Assume image is converted into hr, and lr frequencies.
        n_input = image.shape[0]  # Should return 784, also image is just 1x784
        time = int(time/self.dt)  # Convert to integer
        # Make the spike data.
        m = torch.distributions.Poisson(image)
        spike_times = m.sample(sample_shape=(time,)).long()
        spike_times = torch.clamp(spike_times, max=time-1)
        # Create spikes matrix from spike times.
        spikes = torch.zeros([time, n_input])
        spikes[spike_times[0:time-1, :], :] = 1
        spikes[0, :] = 0
        # Return the input spike occurrence matrix.
        return (spikes, spike_times)

    # def setAssignment(self, label):
    #     # Sets the assignment number for the particular label
    #     self.Assignment[:, label] += self.Activity[:, -1]

    def presentImage(self, image, label, image_duration):

        self.group.v[:] = self.group.Ve
        self.current[:] = 0
        self.CurrCtr[:] = 0
        self.InhibVec[:] = 0
        self.InhibCtr[:] = 0

        spikes, spike_times = self.GenSpkTrain(image, image_duration)
        self.run(spikes, spike_times, image_duration)
        # self.setAssignment(label)
