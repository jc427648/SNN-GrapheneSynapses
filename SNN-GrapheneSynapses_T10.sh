#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=JCU-CL-SNN-GrapheneSynapses_T10
#SBATCH --mail-user=corey.lammie@jcu.edu.au 
#SBATCH --mail-type=END
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=50g
#SBATCH -o logs/T10_out.txt
#SBATCH -e logs/T10_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
module load anaconda/3.6
source activate /scratch/jcu/cl/.conda/memtorch
module load cuda/11.1.1
module load gnu8/8.4.0
module load mvapich2
srun python3 /scratch/jcu/cl/SNN-GrapheneSynapses/Test_10.py