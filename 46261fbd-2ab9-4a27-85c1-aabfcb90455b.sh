#!/bin/bash
#PBS -j oe
#PBS -m ae
#PBS -N SNN-GrapheneSynapses-100
#PBS -o "/scratch/user/benwalters/SNN-GrapheneSynapses/out.txt"
#PBS -e "/scratch/user/benwalters/SNN-GrapheneSynapses/error.txt"
#PBS -M ben.walters@jcu.edu.au
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=10gb
shopt -s expand_aliases
source /etc/profile.d/modules.sh
cd "/scratch/user/benwalters/SNN-GrapheneSynapses"
. ~/.bash_profile
module load anaconda/2020.02
source /sw/RCC/Anaconda/2020.02/etc/profile.d/conda.sh
conda activate /scratch/user/benwalters/conda_env
cd /scratch/user/benwalters/SNN-GrapheneSynapses
python Grid.py --stdpCC 0.0 --stdpDD 0.0
