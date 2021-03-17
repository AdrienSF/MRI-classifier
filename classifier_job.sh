#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0-0:00:01
#SBATCH --job-name=PYTEST
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

python --version