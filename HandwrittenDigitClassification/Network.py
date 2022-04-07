import torch
import os
import sys
from struct import unpack
import numpy as np
import pickle
from STDPsynapses import STDPSynapse, LIFNeuronGroup


def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    if type(sorted_idx) is not np.ndarray:
        sorted_idx = np.array([sorted_idx])
        
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]

class Network:
    def __init__(
        self,
        n_output_neurons=30,
        n_samples_memory=30,
        Ve=0.0095464,
        tau=0.0015744,
        R=499.12,
        gamma=0.029254,
        target_activity=10,
        v_th_min=1e-3,#10e-3
        v_th_max=0.5,#0.029254,
        fixed_inhibition_current=-6.0241e-04,
        dt=2e-4,
        output_dir="output",
    ):
        self.synapse = STDPSynapse(n_output_neurons, wmin=-10e-6, wmax=10e-6)
        self.group = LIFNeuronGroup(
            n_output_neurons,
            Ve=Ve,
            tau=tau,
            R=R,
            gamma=gamma,
            target=target_activity,
            VthMin=v_th_min,
            VthMax=v_th_max,
        )
        self.fixed_inhibition_current = fixed_inhibition_current
        self.dt = dt
        self.output_dir = output_dir
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
        # In this instance, mode is either train or test, and time is the time the image is presented.
        c_p_w = 10 * self.dt  # Current Pulse Width
        i_p_w = 27 * self.dt  # Inhibition Pulse Width
        inhib = self.fixed_inhibition_current  # Fixed inhibition current (A)

        for t in range(int(time / self.dt)):
            # Work out the current, then step the voltages, and then decrement current counters.
            # Spikes is a tensor, not a list.
            self.CurrCtr[spikes[t, :] == 1] = c_p_w
            # Apply inhibition.
            currentMat = torch.multiply(
                self.CurrCtr > 0, self.synapse.w
            )  # Matrix of all values
            self.current = torch.sum(currentMat, 1)
            self.current += self.InhibVec
            # Decrement the current pulse widths.
            self.group.step(self.dt, self.current, self.sumAct, update_parameters)
            self.CurrCtr -= self.dt
            self.InhibCtr -= self.dt
            if torch.sum(self.group.s) > 0:
                if update_parameters:
                    DeltaT = t - spike_times
                    DeltaT = DeltaT * self.dt
                    Neur = torch.unsqueeze(self.group.s, 1)
                    DeltaTP = DeltaT[DeltaT > 0.]
                    if DeltaTP.numel() > 0:
                        DeltaTP = DeltaTP.min(axis=0)[0]
                        integrated_current_p = torch.tensor(self.STDPWindow[1, closest_argmin(DeltaTP.numpy(), self.STDPWindow[0, :])])
                        self.synapse.w += torch.multiply(Neur, integrated_current_p)

                    DeltaTN = DeltaT[DeltaT < 0.]
                    if DeltaTN.numel() > 0:
                        DeltaTN = DeltaTN.max(axis=0)[0]
                        integrated_current_n = torch.tensor(self.STDPWindow[1, closest_argmin(DeltaTN.numpy(), self.STDPWindow[0, :])])
                        self.synapse.w += torch.multiply(Neur, integrated_current_n)
                    
                    self.synapse.w = torch.clamp(self.synapse.w, self.synapse.wmin, self.synapse.wmax)

                # Update activity
                self.Activity[
                    :, self.current_sample
                ] += self.group.s  # Important to track all spiking activity
                self.sumAct = torch.sum(self.Activity, dim=1)
                # Update inhibition
                self.InhibCtr[torch.logical_not(self.group.s)] = i_p_w
                self.InhibVec = torch.multiply(self.InhibCtr > 0, inhib)

    def genSpkTrain(self, image, time):
        # Generate Poissonian spike times to input into the network. Note time is integer.
        # Assume image is converted into hr, and lr frequencies.
        n_input = image.shape[0]  # Should return 784, also image is just 1x784
        time = int(time / self.dt)  # Convert to integer
        # Make the spike data.
        m = torch.distributions.Poisson(image)
        spike_times = torch.cumsum(m.sample(sample_shape=(time,)).long(), dim=0)
        spike_times[spike_times >= time] = 0
        spikes = torch.zeros([time, n_input])
        spikes[spike_times[np.arange(time), 1:], np.arange(n_input)[1:]] = 1
        # Return the input spike occurrence matrix.
        return (spikes, spike_times)

    def setAssignment(self, label):
        # Sets the assignment number for the particular label
        self.Assignment[:, label] += self.Activity[:, self.current_sample]

    def save(self, path="model.pt"):
        d = {}
        d["synapse_w"] = self.synapse.w
        d["group_vth"] = self.group.Vth
        d["assignments"] = self.Assignment
        pickle.dump(d, open(os.path.join(self.output_dir, path), "wb"))

    def load(self, path="model.pt"):
        d = pickle.load(open(os.path.join(self.output_dir, path), "rb"))
        self.synapse.w = d["synapse_w"]
        self.group.Vth = d["group_vth"]
        self.Assignment = d["assignments"]

    def presentImage(self, image, label, image_duration, update_parameters=True):
        self.group.v[:] = self.group.Ve
        self.current[:] = 0
        self.CurrCtr[:] = 0
        self.InhibVec[:] = 0
        self.InhibCtr[:] = 0
        spikes, spike_times = self.genSpkTrain(image, image_duration)
        self.run(
            spikes, spike_times, image_duration, update_parameters=update_parameters
        )
        if update_parameters:
            self.setAssignment(
                label
            )  # You only when update neuron when assignments when training, not testing.

    def detPredictedLabel(self):
        return self.Assignment.max(dim=1)[1][
            torch.max(self.Activity[:, self.current_sample], 0, keepdims=True)[1].item()
        ].item()

    def OverwriteActivity(self):
        # This function will overWrite the activity from n_memory_samples ago.
        self.sumAct -= self.Activity[:, self.current_sample]
        self.Activity[:, self.current_sample] = torch.zeros(self.n_output_neurons)

    def UpdateCurrentSample(self):
        self.current_sample += 1
        if self.current_sample == self.n_samples_memory:
            self.current_sample = 0
