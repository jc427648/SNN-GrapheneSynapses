#!/bin/bash
#PBS -j oe
#PBS -m ae
#PBS -N SNN-GrapheneSynapses-Eval
#PBS -o "/home/jc299170/Evaluate/SNN-GrapheneSynapses/Handwritten Digit Classification/30_eval.log"
#PBS -e "/home/jc299170/Evaluate/SNN-GrapheneSynapses/Handwritten Digit Classification/30_eval.log"
#PBS -M corey.lammie@jcu.edu.au
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=10:mem=100gb

shopt -s expand_aliases
source /etc/profile.d/modules.sh
cd "/home/jc299170/Evaluate/SNN-GrapheneSynapses/Handwritten Digit Classification"
. ~/.bash_profile
module load conda3
source $CONDA_PROF/conda.sh
conda activate base
python Evaluate_30.py