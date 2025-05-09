#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=default
#SBATCH --gres=gpu:1

source /fs/clip-scratch/nehasrik/init_conda.sh
module load cuda
cd /fs/clip-scratch/nehasrik/paraphrase-nlu
conda activate para-nlu

python -m data_selection.defeasible.select