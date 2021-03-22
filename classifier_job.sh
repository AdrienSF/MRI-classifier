#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0-0:10:00
#SBATCH --job-name=crossVal
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load anaconda3/current
conda init bash
conda activate slurmenv

# python3 pca_validation.py