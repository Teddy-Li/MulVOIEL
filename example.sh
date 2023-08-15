#!/bin/bash

#SBATCH --job-name=JOBNAME
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=ec216

module load python/3.10.8-gpu

CONDA_ROOT=/work/ec216/ec216/teddy/condaenvs/
export CONDARC=${CONDA_ROOT}/.condarc
eval "$(conda shell.bash hook)"

conda activate work

cp -r SOURCE_DIR /work/ec216/ec216/teddy/TGT_DIR

# YOUR SCRIPT HERE

cp -r /work/ec216/ec216/teddy/TGT_DIR SOURCE_DIR
rm -r /work/ec216/ec216/teddy/TGT_DIR