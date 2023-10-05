#!/bin/bash -login
#SBATCH -p high2  
#SBATCH -J VP_ML            
#SBATCH -c 32
#SBATCH -A datalabgrp
#SBATCH -t 1-11:00:00    
#SBATCH --mem=64000            
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsliu@ucdavis.edu
#SBATCH --output=full_genome_CNN_CUDA_drop0.4_1002_am.log

# initialize conda
. ~/miniconda3/etc/profile.d/conda.sh

# activate your desired conda environment
conda activate pytorch

# fail on weird errors
set -e
set -x

### YOUR COMMANDS GO HERE ###
python CNN.py /home/zhuosl/VPML/Genome_matrix/genome_matrix.csv 

# Print out values of the current jobs SLURM environment variables
env | grep SLURM

# Print out final statistics about resource use before job exits
scontrol show job ${SLURM_JOB_ID}

sstat --format 'JobID,MaxRSS,AveCPU' -j ${SLURM_JOB_ID}