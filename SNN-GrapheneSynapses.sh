#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=JCU-CL-SNN-GrapheneSynapses
#SBATCH --mail-user=corey.lammie@jcu.edu.au 
#SBATCH --mail-type=END
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=50g
#SBATCH -o logs/JCU-CL-SNN-GrapheneSynapses_out.txt
#SBATCH -e logs/JCU-CL-SNN-GrapheneSynapses_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
module load anaconda/3.6
source activate /scratch/jcu/cl/.conda/memtorch
module load cuda/11.1.1
module load gnu8/8.4.0
module load mvapich2
srun python3 /scratch/jcu/cl/SNN-GrapheneSynapses/training.py