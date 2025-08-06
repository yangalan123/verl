#!/bin/bash

#SBATCH --chdir=/fsx/zhuokai/verl/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=verl_demo_annealed_sampling
#SBATCH --output=/fsx/zhuokai/verl/slurm/verl_demo_annealed_sampling.stdout
#SBATCH --error=/fsx/zhuokai/verl/slurm/verl_demo_annealed_sampling.stderr

bash recipe/annealed_sampling/demo_verl_annealed_sampling.sh