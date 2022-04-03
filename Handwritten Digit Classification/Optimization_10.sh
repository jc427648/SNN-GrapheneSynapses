#!/bin/bash
#PBS -j oe
#PBS -m ae
#PBS -N SNN-GrapheneSynapses
#PBS -o "/home/jc299170/SNN-GrapheneSynapses/Handwritten Digit Classification/10.log"
#PBS -e "/home/jc299170/SNN-GrapheneSynapses/Handwritten Digit Classification/10.log"
#PBS -M corey.lammie@jcu.edu.au
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=10gb

shopt -s expand_aliases
source /etc/profile.d/modules.sh
cd "/home/jc299170/SNN-GrapheneSynapses/Handwritten Digit Classification"
. ~/.bash_profile
module load conda3
source $CONDA_PROF/conda.sh
conda activate base
python Optimization_10.py