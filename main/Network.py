import torch
import os
import sys
from struct import unpack
import numpy as np
from STDPsynapses import STDPSynapse, LIFNeuronGroup


class Network():
    # Define the network. Might need super()
    def __init__(self, n, mode, dt=0.2e-3):
        # n is the number of neurons.
        self.synapse = STDPSynapse(n, wmin=-45e-3, wmax=45e-3)
        self.group = LIFNeuronGroup(
            n, mode, Ve=0.0, tau=0.1, R=1000, gamma=0.005, target=10, VthMin=0.25, VthMax=50)
        self.dt = dt
        self.n = n
        self.Activity = torch.zeros_like(torch.Tensor(n, 3 * n))
        self.sumAct = torch.zeros(n)
        self.STDPWindow = self.synapse.GetSTDP()
        self.Assignment = torch.zeros((n, 10))
        # Current Counter (not correct, should be a vector)
        self.CurrCtr = torch.zeros((784))
        # Inhibition counter (May not be needed)
        self.InhibCtr = torch.zeros((self.n))
        self.InhibVec = torch.zeros((self.n))
        self.current = torch.zeros((self.n))

    def run(self, mode, spikes, spike_times, time, PatCount):
        # in this instance, mode is either train or test, and time is the time the image is presented.
        # My thinking is this, have another method dedicated to getting the MNIST and generating the spike train.
        # Then do the steps for all fo the time range, make sure to log the info to get idea of time.

        # Current should be a nx1 vector of the sum of all currents (including inhibition.)
        # Consider having creating one more file to run the entire MNIST simulation.
        c_p_w = 1e-3  # Current Pulse Width (s)
        i_p_w = 1e-3  # Inhibition Pulse Width (s)
        inhib = -1.0  # Fixed inhibition current (A)

        for t in range(int(time / self.dt)):
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
                    DeltaT = t - spike_times
                    # Ensure time is 80ms long (off STDP)
                    DeltaT[DeltaT == t] = 80e-3 / self.dt

                    DeltaTP, indices = torch.where(DeltaT > 0, DeltaT, 400 * torch.ones(
                        784, dtype=torch.int)).min(axis=0)  # 400 should be some variable
                    DeltaTP = 1e3 * self.dt * DeltaTP
                    DeltaTN, indices = torch.where(
                        DeltaT < 0, DeltaT, -400 * torch.ones(784, dtype=torch.int)).max(axis=0)
                    DeltaTN = 1e3 * self.dt * DeltaTN

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
        self.Activity[:, PatCount] = torch.zeros(self.n)

    def GetMNIST(self, N, mode):
        # Get the MNIST images and their corresponding labels as a series of tuples.STDPsynapse N is number of images.

        if (mode == 'train'):
            Imgs = open('train-images.idx3-ubyte', 'rb')
            Lbls = open('train-labels.idx1-ubyte', 'rb')

            Imgs.read(4)  # Magic number
            NumImages = unpack('>I', Imgs.read(4))[0]
            rows = unpack('>I', Imgs.read(4))[0]
            cols = unpack('>I', Imgs.read(4))[0]

            Lbls.read(4)  # magic number
            NumLabels = unpack('>I', Lbls.read(4))[0]

        elif (mode == 'test'):

            Imgs = open('t10k-images.idx3-ubyte', 'rb')
            Lbls = open('t10k-labels.idx1-ubyte', 'rb')

            Imgs.read(4)
            NumImages = unpack('>I', Imgs.read(4))[0]
            rows = unpack('>I', Imgs.read(4))[0]
            cols = unpack('>I', Imgs.read(4))[0]

            Lbls.read(4)
            NumLabels = unpack('>I', Lbls.read(4))[0]

        if (NumImages != NumLabels):
            raise Exception('Number of labels did not match number of images')

        X = np.zeros((N, rows, cols), dtype=np.uint8)  # Store all images
        y = np.zeros((N, 1), dtype=np.uint8)

        for i in range(N):
            if (i % 1000 == 0):
                print('Progress :', i, '/', N)
            X[i] = [[unpack('>B', Imgs.read(1))[0] for unused_col in
                     range(cols)] for unused_row in range(rows)]
            y[i] = unpack('>B', Lbls.read(1))[0]
        print('Progress :', N, '/', N, '\n')

        # These values are in between 0 and 255, but in should be 20 and 200.Need to fix
        X = X.reshape([N, 784])
        # X = X/255*180+20 #I may have misinterpreted the MNIST classification
        UpperFreq = 200
        LowerFreq = 20
        UpperTime = 1 / UpperFreq
        LowerTime = 1 / LowerFreq
        #ValEnd = LowerTime/self.dt
        #ValMultiplied = (UpperTime-LowerTime)/self.dt
        # X = X/255*(UpperTime-LowerTime)/self.dt  + LowerTime/self.dt #Incorrect, np.random uses the time not the frequency.
        # When x is 1, freq = 200, time = 0.005. When x is 0, freq = 20,time = 0.05
        # It's also been found that spike times might the use index and not actual value. Need to reflect in choice of frequencies.
        # Converting to binary image
        level = 50  # Level above which conversion occurs
        X = np.where(X < level, LowerTime / self.dt, UpperTime / self.dt)

        # Store img and lbl in dictionary for reference.
        data = {'X': X, 'y': y}

        return data

    def GenSpkTrain(self, image, time):
        # Generate Poissonian spike times to input into the network. Note time is integer.
        # Assume image is converted into hr, and lr frequencies.Exception()

        n_input = image.shape[0]  # Should return 784, also image is just 1x784
        time = int(time / self.dt)  # Convert to integer
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
