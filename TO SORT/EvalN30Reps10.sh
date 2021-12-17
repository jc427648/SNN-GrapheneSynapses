#!/bin/bash
### qstat -f -Q
#PBS -N SNN-GrapheneSynapses_evaluate
#PBS -P JCU-SNN
#PBS -o /scratch/user/benwalters/EvalN30Reps10_out.txt
#PBS -e /scratch/user/benwalters/EvalN30Reps10_error.txt
#PBS -q Single
#PBS -l walltime=40:00:00
#PBS -l select=1:mem=10gb:ncpus=4:mpiprocs=4
#PBS -m abe -M ben.walters@my.jcu.edu.au

module load anaconda/2020.02
source /sw/RCC/Anaconda/2020.02/etc/profile.d/conda.sh
conda activate /scratch/user/benwalters/conda_env
cd /scratch/user/benwalters/SNN-GrapheneSynapses/
python EvalN30Reps10.py