#!/bin/bash


#SBATCH --mem=256000
#SBATCH --nodes=6
#SBATCH --time=0-10:00:00
#SBATCH --job-name=shaplot
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load anaconda3/current
eval "$(conda shell.bash hook)"
conda init bash
conda activate shapenv

python3 linear_svm_crossval.py
