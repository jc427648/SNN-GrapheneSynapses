#!/bin/bash
### qstat -f -Q
#PBS -N SNN-GrapheneSynapses_Optimization_300
#PBS -P JCU-SNN
#PBS -o /scratch/user/coreylammie/SNN-GrapheneSynapses/logs/300_out.txt
#PBS -e /scratch/user/coreylammie/SNN-GrapheneSynapses/logs/300_error.txt
#PBS -q Single
#PBS -l walltime=167:00:00
#PBS -l select=1:mem=10gb:ncpus=8:mpiprocs=8
#PBS -m abe -M corey.lammie@jcu.edu.au

module load anaconda/2020.02
source /sw/RCC/Anaconda/2020.02/etc/profile.d/conda.sh
conda activate /scratch/user/coreylammie/conda_env
cd /scratch/user/coreylammie/SNN-GrapheneSynapses/
python Evaluate_300.py