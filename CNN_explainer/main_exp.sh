#!/bin/bash -login
#SBATCH -p gpu-a100-h         
#SBATCH -J VP_ML            
#SBATCH -c 32
#SBATCH -A datalabgrp
#SBATCH -t 1-11:00:00    
#SBATCH --mem=64000            
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsliu@ucdavis.edu
#SBATCH --gres=gpu:1

# initialize conda
. ~/miniconda3/etc/profile.d/conda.sh

# activate your desired conda environment
conda activate pytorch

# fail on weird errors
set -e
set -x

# Define an array of your matrix filenames
MATRIX_FILES=("genome_matrix_full.csv")

# Directory where your matrix files are stored
MATRIX_DIR="/home/zhuosl/VPML/Genome_matrix"

for MATRIX_FILE in "${MATRIX_FILES[@]}"
do
    echo "Processing $MATRIX_FILE"
    python main_exp.py "${MATRIX_DIR}/${MATRIX_FILE}" 
done

# Print out values of the current jobs SLURM environment variables
env | grep SLURM

# Print out final statistics about resource use before job exits
scontrol show job ${SLURM_JOB_ID}

sstat --format 'JobID,MaxRSS,AveCPU' -j ${SLURM_JOB_ID}