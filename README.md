# Unsupervised Character Recognition with Graphene Memristive Synapses
Supplimentary GitHub Repository containing source codes for the paper entitled *Unsupervised Character Recognition with Graphene Memristive Synapses*, which is currently under consideration for publication in Advanced Intelligent Systems (AISY), as an interactive paper.

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

Many scripts were developed for execution on a High Performance Computing (HPC) clsuter using `SLURM`. Consequently, for many scripts, we provide bash scripts which can be submitted using `SLURM` with some modification.
