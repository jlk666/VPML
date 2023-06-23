#!/bin/bash -login
#SBATCH -p med2                # use 'med2' partition for medium priority
#SBATCH -J myjob               # name for job
#SBATCH -c 1                   # 1 core
#SBATCH -t 5:00:00             # ask for an hour, max
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsliu@ucdavis.edu
#SBATCH -e Genome_catcher.j%j.err                   # STANDARD ERROR FILE TO WRITE TO
#SBATCH -o Genome_catcher.j%j.out                   # STANDARD OUTPUT FILE TO WRITE TO


# activate your desired conda environment
conda activate VPML

# fail on weird errors
set -e


###  COMMANDS GO HERE ###
python Genome_catcher.py
###  COMMANDS GO HERE ###

# Print out values of the current jobs SLURM environment variables
env | grep SLURM

# Print out final statistics about resource use before job exits
scontrol show job ${SLURM_JOB_ID}

sstat --format 'JobID,MaxRSS,AveCPU' -j ${SLURM_JOB_ID}