# Unsupervised Character Recognition with Graphene Memristive Synapses
Supplimentary GitHub Repository containing source codes for the paper entitled *Unsupervised Character Recognition with Graphene Memristive Synapses*, which is currently under consideration for publication in Neural Computing and Architectures.

## Abstract
Memristive devices being applied in neuromorphic computing are envisioned to significantly improve the power consumption and
speed of future computing platforms. The materials used to fabricate such devices will play a significant role in their viability.
Graphene is a promising material, with superb electrical properties and the ability to be produced in large volumes sustainably. In
this paper, we demonstrate that graphene-pentacene devices can be used as synapses within SNNs to realise Spike Timing Dependant
Plasticity (STDP) for unsupervised learning in an efficient manner. Specifically, we verify operation of two SNN architectures tasked for
single digit (0-9) classification: (i) a single layer network, where inputs are presented in 5x5 pixel resolution, and (ii) a larger network
capable of classifying the Modified National Institute of Standards and Technology (MNIST) dataset, where inputs are presented in
28x28 pixel resolution. Final results demonstrate that for 100 output neurons, after one training epoch, a test set accuracy of up
to 86% can be achieved, which is higher than prior art using the same number of output neurons. We attribute this performance
improvement to homeostatic plasticity dynamics that we used to alter the threshold of neurons during training. Our work presents the
first investigation of the use of green-fabricated graphene memristive devices to perform a complex pattern classification task. This
can pave the way for future research in using graphene devices with memristive capabilities in neuromorphic computing architectures..

## Requirements
All scripts are developed using `MATLAB` and `Python`. To run all scripts, a `MATLAB` installation of 2020 or newer is required, and a working `Python` 3.7 or newer runtime is required.
All Python requirements can be installed using the following command:

```
pip install -r requirements.txt
```

Many scripts were developed for execution on a High Performance Computing (HPC) clsuter using `SLURM`. Consequently, for many scripts, parallelization is used.

## Documentation
This GitHub Repository is comprised of three folders: `Experimental Data`, `Handwritten Digit Classification`, `Interactive Figures`, and `Unsupervised Binary Pattern Classification`.

### Experimental Data
Contains all experimental data (`pentacenesingle200_2_slow_k2400.txt`, `pentacenesingle200_3_slow_k2400.txt`, `pentacenesingle200_4_slow_k2400.txt`, and `pentacenesingle200_slow_k2400.txt`), the VTEAM model fitting script (`VTEAM_fit.m`), and a script used to generate the STDP window used in our simulations (`gen_STDP_window.m`).

### Handwritten Digit Classification
Contains all scripts (`Main.py`, `MNISTDataLoader.py`, `Network.py`, `Plotting.py`, `set_all_seeds.py`, and `STDPsynapses.py`) required to train and evaluate our simulated SNN architectures. `Evaluate.py` can be used to evaluate different network architectures using the parameters reported in Table 2 in conjunction with `Evaluate_10.py`, `Evaluate_30.py`, `Evaluate_100.py`, `Evaluate_300.py`, and `Evaluate_500.py`, for 10, 30, 100, 300, and 500 neuron configurations, respectively.

### Interactive Figures
Contains all interactive figures embedded using Authorea. Files are named using `Figure_X.py`, where `X` is the corresponding figure number. All interactive figure generation scripts produce a svg and `html` `file`.

### Unsupervised Binary Pattern Classification
Contains all scripts used to perform binary pattern classification. `Main.m` can be used to regenerate the receptive field which has been reported in our paper.
