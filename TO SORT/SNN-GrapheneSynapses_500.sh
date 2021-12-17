#!/bin/bash
### qstat -f -Q
#PBS -N SNN-GrapheneSynapses_Optimization_500
#PBS -P JCU-SNN
#PBS -o /scratch/user/coreylammie/SNN-GrapheneSynapses/logs/500_out.txt
#PBS -e /scratch/user/coreylammie/SNN-GrapheneSynapses/logs/500_error.txt
#PBS -q Short
#PBS -l walltime=24:00:00
#PBS -l select=1:mem=10gb:ncpus=8:mpiprocs=8
#PBS -m abe -M corey.lammie@jcu.edu.au

module load anaconda/2020.02
source /sw/RCC/Anaconda/2020.02/etc/profile.d/conda.sh
conda activate /scratch/user/coreylammie/conda_env
cd /scratch/user/coreylammie/SNN-GrapheneSynapses/
python Evaluate_500.py