#!/bin/bash -login
#SBATCH -p med2                # use 'med2' partition for medium priority
#SBATCH -J myjob               # name for job
#SBATCH -c 4                   # 4 core
#SBATCH -t 00:08:00             # ask for an hour, max
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsliu@ucdavis.edu
#SBATCH -e Genome_catcher.j%j.err                   # STANDARD ERROR FILE TO WRITE TO
#SBATCH -o Genome_catcher.j%j.out                   # STANDARD OUTPUT FILE TO WRITE TO

# initialize conda
. ~/miniconda3/etc/profile.d/conda.sh

# activate your desired conda environment
conda activate VPML

# fail on weird errors
set -e


###  COMMANDS GO HERE ###
snakemake --cores 4 download_genome
###  COMMANDS GO HERE ###

# Print out values of the current jobs SLURM environment variables
env | grep SLURM

# Print out final statistics about resource use before job exits
scontrol show job ${SLURM_JOB_ID}

sstat --format 'JobID,MaxRSS,AveCPU' -j ${SLURM_JOB_ID}