#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --nodelist=clip02

module load cuda
source /fs/clip-projects/rlab/nehasrik/init_conda.sh
conda activate para-nlu
cd /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/roberta/defeasible
export TASK_NAME="mrpc"
export CACHE_DIR='/fs/clip-scratch/nehasrik/paraphrase-nlu/cache'

split='snli'
model='roberta-base'
project_root='/fs/clip-projects/rlab/nehasrik/paraphrase-nlu'

python run_glue.py \
	--model_name_or_path $model \
	--tokenizer_name $model \
	--use_fast_tokenizer false \
	--cache_dir $CACHE_DIR \
	--do_train \
	--train_file $project_root/data_selection/defeasible/$split/analysis_model_examples/train_examples.csv \
	--do_eval \
	--validation_file $project_root/data_selection/defeasible/$split/analysis_model_examples/dev_examples.csv \
	--max_seq_length 128 \
	--per_device_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--seed 42 \
	--save_strategy epoch \
	--learning_rate 5e-6 \
	--num_train_epochs 2 \
	--overwrite_output_dir \
	--output_dir $project_root/modeling/roberta/defeasible/chkpts/analysis_models/d-$split-$model

# TRAINING LARGE MODELS: lr=5e-6, seed=42, batch_size=16, epochs=2
# Base Models: lr=5e-6, seed=42, batch_size=64, epochs=2

#   --gradient_accumulation_steps 8 \
#	--do_lower_case \
# 	--output_attn \
#	--task_name $TASK_NAME \