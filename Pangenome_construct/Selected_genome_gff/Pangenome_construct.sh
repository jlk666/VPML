#!/bin/bash -login
#SBATCH -p high2                # use 'med2' partition for medium priority
#SBATCH -J VP_genome_pangenome_construct             # name for job
#SBATCH -c 32                   
#SBATCH -t 3-12:00:00    
#SBATCH --mem=64000            
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsliu@ucdavis.edu

# initialize conda
. ~/miniconda3/etc/profile.d/conda.sh

# activate your desired conda environment
conda activate pangenome

# fail on weird errors
set -e
set -x

### YOUR COMMANDS GO HERE ###
# for example,
roary -e --mafft -p 8 *.gff
### YOUR COMMANDS GO HERE ###

# Print out values of the current jobs SLURM environment variables
env | grep SLURM

# Print out final statistics about resource use before job exits
scontrol show job ${SLURM_JOB_ID}

sstat --format 'JobID,MaxRSS,AveCPU' -j ${SLURM_JOB_ID}