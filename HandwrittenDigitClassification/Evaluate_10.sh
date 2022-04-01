#!/bin/bash
### qstat -f -Q
#PBS -N SNN-GrapheneSynapses_evaluate
#PBS -P JCU-SNN
#PBS -o /scratch/user/benwalters/ID2_EvalN10_out.txt
#PBS -e /scratch/user/benwalters/ID2s_EvalN10_error.txt
#PBS -q workq
#PBS -l walltime=10:00:00
#PBS -l select=1:mem=10gb:ncpus=4:mpiprocs=4
#PBS -m abe -M ben.walters@my.jcu.edu.au

module load anaconda/2020.02
source /sw/RCC/Anaconda/2020.02/etc/profile.d/conda.sh
conda activate /scratch/user/benwalters/conda_env
cd /scratch/user/benwalters/SNN-GrapheneSynapses/HandwrittenDigitClassification/
python Evaluate_10.py