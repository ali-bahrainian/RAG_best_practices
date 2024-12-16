#!/bin/bash
#SBATCH --job-name='project'
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-fat-galvani
#SBATCH --time=3-00:00  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=outputs.out
#SBATCH --error=outputs.err

# Set Conda environment variables
export XDG_CACHE_HOME='/mnt/qb/work/eickhoff/esx400/.conda/pkgs/cache'
export CONDA_PKGS_DIRS='/mnt/qb/work/eickhoff/esx400/.conda/pkgs/cache'

# Use the Conda environment libraries
export LD_LIBRARY_PATH=/mnt/qb/work/eickhoff/esx400/.conda/ralm/lib:$LD_LIBRARY_PATH


source ~/.bashrc
conda activate $WORK/.conda/ralm

export PATH=/mnt/qb/work/eickhoff/esx400/.conda/ralm/bin:$PATH
source scl_source enable gcc-toolset-11
gcc --version

srun python3 evaluation.py \
    --output-dir /mnt/qb/work/eickhoff/esx400/ralm/project/outputs  