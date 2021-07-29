#!/bin/bash
### qstat -f -Q
#PBS -N SNN-GrapheneSynapses_MNISTDataLoader
#PBS -P JCU-SNN
#PBS -o /scratch/user/coreylammie/SNN-GrapheneSynapses/logs/SNN-GrapheneSynapses_MNISTDataLoader_out.txt
#PBS -e /scratch/user/coreylammie/SNN-GrapheneSynapses/logs/SNN-GrapheneSynapses_MNISTDataLoader_error.txt
#PBS -q Short
#PBS -l walltime=2:00:00
#PBS -l select=1:mem=1gb:ncpus=1:mpiprocs=1
#PBS -m abe
#PBS -M corey.lammie@jcu.edu.au

module load anaconda/2020.02
source /sw/RCC/Anaconda/2020.02/etc/profile.d/conda.sh
conda activate /scratch/user/coreylammie/conda_env
cd /scratch/user/coreylammie/SNN-GrapheneSynapses/
python MNISTDataLoader.py
