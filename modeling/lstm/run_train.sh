#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:1

module load cuda
source /fs/clip-scratch/nehasrik/init_conda.sh
conda activate infersent
cd /fs/clip-scratch/nehasrik/paraphrase-nlu/modeling/lstm

python train_nli.py \
    --word_emb_path 'dataset/GloVe/glove.840B.300d.txt' \
    --nlipath 'dataset/anli/' \
    --outputdir 'dataset/anli/infersent/' \
    --outputmodelname 'infersent.pth' \
    --optimizer 'sgd' \
    --nonlinear_fc 0 \
