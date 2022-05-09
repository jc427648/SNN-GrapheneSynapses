#!/bin/bash
### qstat -f -Q
#PBS -N SNN-GrapheneSynapses_evaluate
#PBS -P JCU-SNN
#PBS -o /scratch/user/benwalters/10Trial_EvalN100_out.txt
#PBS -e /scratch/user/benwalters/10Trial_EvalN100_error.txt
#PBS -q workq
#PBS -l walltime=80:00:00
#PBS -l select=1:mem=10gb:ncpus=10:mpiprocs=10
#PBS -m abe -M ben.walters@my.jcu.edu.au

module load anaconda/2020.02
source /sw/RCC/Anaconda/2020.02/etc/profile.d/conda.sh
conda activate /scratch/user/benwalters/conda_env
cd /scratch/user/benwalters/SNN-GrapheneSynapses
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/user/benwalters/conda_env/lib
python Evaluate_100.py