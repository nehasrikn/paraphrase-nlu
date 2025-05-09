#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000
#SBATCH --mem=32g

module load cuda
source /fs/clip-projects/rlab/nehasrik/init_conda.sh
conda activate a6000

export CACHE_DIR='/fs/clip-scratch/nehasrik/paraphrase-nlu/cache'

python /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/roberta/abductive/run_multiple_choice.py \
   	--model_name_or_path microsoft/deberta-v3-large \
	--tokenizer_name microsoft/deberta-v3-large \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir /fs/clip-scratch/nehasrik/paraphrase-nlu/deberta-models/anli-deberta-v3-large \
	--max_seq_length 128 \
	--per_device_train_batch_size=8 \
	--per_gpu_eval_batch_size=8 \
    --learning_rate 5e-6 \
	--num_train_epochs 2 \
	--warmup_steps 50 \
    --save_strategy epoch \
    --seed 42 \
    --cache_dir $CACHE_DIR \