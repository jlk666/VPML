#!/bin/bash -login
#SBATCH -p high2          
#SBATCH -J VP_ML            
#SBATCH -c 32
#SBATCH -A datalabgrp
#SBATCH -t 3-12:00:00    
#SBATCH --mem=64000            
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsliu@ucdavis.edu

# initialize conda
. ~/miniconda3/etc/profile.d/conda.sh

# activate your desired conda environment
conda activate sklearn-env

# fail on weird errors
set -e
set -x

### YOUR COMMANDS GO HERE ###
# Define an array of your matrix filenames
MATRIX_FILES=("core_genome.csv" "core_shell_genome.csv" "core_soft_genome.csv" "genome_matrix_full.csv")

# Directory where your matrix files are stored
MATRIX_DIR="/home/zhuosl/VPML/Genome_matrix"

# Loop through each file and execute your Python script
for MATRIX_FILE in "${MATRIX_FILES[@]}"
do
    echo "Processing $MATRIX_FILE"
    python MLScript.py "${MATRIX_DIR}/${MATRIX_FILE}" ALL
done
### YOUR COMMANDS GO HERE ###

# Print out values of the current jobs SLURM environment variables
env | grep SLURM

# Print out final statistics about resource use before job exits
scontrol show job ${SLURM_JOB_ID}

sstat --format 'JobID,MaxRSS,AveCPU' -j ${SLURM_JOB_ID}
