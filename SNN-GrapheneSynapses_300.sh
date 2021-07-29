#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=JCU-CL-SNN-GrapheneSynapses_300
#SBATCH --mail-user=corey.lammie@jcu.edu.au 
#SBATCH --mail-type=END
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=50g
#SBATCH -o logs/300_out.txt
#SBATCH -e logs/300_error.txt
module load anaconda/3.6
source activate /scratch/jcu/cl/.conda/memtorch
module load gnu8/8.4.0
srun python3 /scratch/jcu/cl/SNN-GrapheneSynapses/Optimization_300.py