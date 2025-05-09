#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:1

module load cuda
source /fs/clip-scratch/nehasrik/init_conda.sh
conda activate para-nlu
cd /fs/clip-scratch/nehasrik/paraphrase-nlu/modeling/defeasible
export TASK_NAME="mrpc"
export CACHE_DIR='/fs/clip-scratch/nehasrik/paraphrase-nlu/modeling/hf-cache'
python run_glue.py \
	--model_name_or_path roberta-base \
	--tokenizer_name roberta-base \
	--use_fast_tokenizer false \
	--cache_dir $CACHE_DIR \
	--do_train \
	--train_file /fs/clip-scratch/nehasrik/paraphrase-nlu/data_selection/defeasible/social/aflite_embedding_model_examples/aflite_train.csv \
	--do_eval \
	--validation_file /fs/clip-scratch/nehasrik/paraphrase-nlu/data_selection/defeasible/social/aflite_embedding_model_examples/aflite_dev.csv \
	--max_seq_length 128 \
	--per_device_train_batch_size=32 \
	--per_gpu_eval_batch_size=32 \
	--seed 42 \
	--save_strategy epoch \
	--learning_rate 2e-5 \
	--num_train_epochs 2 \
	--overwrite_output_dir \
	--output_dir chkpts/aflite_embedding_models/d-social-roberta-base

#   --gradient_accumulation_steps 8 \
#	--do_lower_case \
# 	--output_attn \
#	--task_name $TASK_NAME \