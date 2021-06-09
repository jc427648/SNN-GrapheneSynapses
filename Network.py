import torch
import os
import sys
from struct import unpack
import numpy as np
from STDPsynapses import STDPSynapse, LIFNeuronGroup


class Network():
    def __init__(self, n_output_neurons=30, dt=0.2e-3):
        self.synapse = STDPSynapse(n_output_neurons, wmin=-45e-3, wmax=45e-3)
        self.group = LIFNeuronGroup(
            n_output_neurons, Ve=0.0, tau=0.1, R=1000, gamma=0.005, target=10, VthMin=0.25, VthMax=50)
        self.dt = dt
        self.n_output_neurons_output_neurons = n_output_neurons
        self.Activity = torch.zeros_like(
            torch.Tensor(n_output_neurons, 3*n_output_neurons))
        self.sumAct = torch.zeros(n_output_neurons)
        self.STDPWindow = self.synapse.GetSTDP()
        self.Assignment = torch.zeros((n_output_neurons, 10))
        # Current Counter (not correct, should be a vector)
        self.CurrCtr = torch.zeros((784))
        # Inhibition counter (May not be needed)
        self.InhibCtr = torch.zeros((self.n_output_neurons))
        self.InhibVec = torch.zeros((self.n_output_neurons))
        self.current = torch.zeros((self.n_output_neurons))

    def run(self, mode, spikes, spike_times, time, PatCount):
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

            currentMat = torch.multiply(
                self.CurrCtr > 0, self.synapse.w)  # Matrix of all values
            self.current = torch.sum(currentMat, 1)
            self.current += self.InhibVec

            # Decrement the current pulse widths.
            self.group.step(self.dt, self.current, self.sumAct, mode)
            self.CurrCtr -= self.dt
            self.InhibCtr -= self.dt

            if torch.sum(self.group.s) > 0:
                if (mode == 'train'):

                    # Update synaptic weights
                    DeltaT = t-spike_times
                    # Ensure time is 80ms long (off STDP)
                    DeltaT[DeltaT == t] = 80e-3/self.dt

                    DeltaTP, indices = torch.where(DeltaT > 0, DeltaT, 400*torch.ones(
                        784, dtype=torch.int)).min(axis=0)  # 400 should be some variable
                    DeltaTP = 1e3*self.dt*DeltaTP
                    DeltaTN, indices = torch.where(
                        DeltaT < 0, DeltaT, -400*torch.ones(784, dtype=torch.int)).max(axis=0)
                    DeltaTN = 1e3*self.dt*DeltaTN

                    Neur = torch.unsqueeze(self.group.s, 1)

                    self.synapse.potentiate(DeltaTP, Neur, self.STDPWindow)
                    self.synapse.depress(DeltaTN, Neur, self.STDPWindow)

                # Update activity counters
                self.Activity[:, PatCount] += self.group.s
                self.sumAct += self.group.s

                # Update inhibition
                self.InhibCtr[torch.logical_not(self.group.s)] = i_p_w
                self.InhibVec = torch.multiply(self.InhibCtr > 0, inhib)

    def resetActivity(self, PatCount):
        # Overwrite the pattern activity from n patterns ago
        self.sumAct -= self.Activity[:, PatCount]
        self.Activity[:, PatCount] = torch.zeros(self.n_output_neurons)

    def GenSpkTrain(self, image, time):
        # Generate Poissonian spike times to input into the network. Note time is integer.
        # Assume image is converted into hr, and lr frequencies.Exception()

        n_input = image.shape[0]  # Should return 784, also image is just 1x784
        time = int(time/self.dt)  # Convert to integer
        # Make the spike data.
        spike_times = np.random.poisson(image, [time, n_input])
        spike_times = np.cumsum(spike_times, axis=0)
        spike_times[spike_times >= time] = 0

        # Create spikes matrix from spike times.
        spikes = np.zeros([time, n_input])
        for idx in range(time):
            spikes[spike_times[idx, :], np.arange(n_input)] = 1

        # Temporary fix: The above code forces a spike from
        # every input neuron on the first time step.
        spikes[0, :] = 0

        # Return the input spike occurrence matrix.
        return (torch.from_numpy(spikes), torch.from_numpy(spike_times))

    def setAssignment(self, label, PatCount):
        # Sets the assignment number for the particular label
        self.Assignment[:, label] += self.Activity[:, PatCount]

    def loadTrainedParameters(self):
        pass
