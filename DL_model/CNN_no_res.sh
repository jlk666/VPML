#!/bin/bash -login
#SBATCH -p gpu-a100-h         
#SBATCH -J VP_ML            
#SBATCH -c 32
#SBATCH -A datalabgrp
#SBATCH -t 3-12:00:00    
#SBATCH --mem=64000            
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsliu@ucdavis.edu
#SBATCH --output=filter_genome_CNN_nores_Average_pooling_probdrop_80_CUDA.log

# initialize conda
. ~/miniconda3/etc/profile.d/conda.sh

# activate your desired conda environment
conda activate pytorch

# fail on weird errors
set -e
set -x

### YOUR COMMANDS GO HERE ###
python CNN_no_res.py /home/zhuosl/VPML/Genome_matrix/genome_matrix.csv 

# Print out values of the current jobs SLURM environment variables
env | grep SLURM

# Print out final statistics about resource use before job exits
scontrol show job ${SLURM_JOB_ID}

sstat --format 'JobID,MaxRSS,AveCPU' -j ${SLURM_JOB_ID}