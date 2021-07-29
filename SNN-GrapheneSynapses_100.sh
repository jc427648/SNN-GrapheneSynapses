#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=JCU-CL-SNN-GrapheneSynapses_100
#SBATCH --mail-user=corey.lammie@jcu.edu.au 
#SBATCH --mail-type=END
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=50g
#SBATCH -o logs/100_out.txt
#SBATCH -e logs/100_error.txt
module load anaconda/3.6
source activate /scratch/jcu/cl/.conda/memtorch
module load cuda/11.1.1
module load gnu8/8.4.0
module load mvapich2
srun python3 /scratch/jcu/cl/SNN-GrapheneSynapses/Optimization_100.py