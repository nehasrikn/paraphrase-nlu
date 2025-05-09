#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000
#SBATCH --mem=32g

module load cuda
source /fs/clip-projects/rlab/nehasrik/init_conda.sh
conda activate a6000

# cd /fs/clip-projects/rlab/nehasrik/miniconda3/envs/a6000/lib/python3.8/site-packages/torch/lib
# ln -s libnvrtc-672ee683.so.11.2 libnvrtc.so


# cd /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/roberta/defeasible
export TASK_NAME="mrpc"
export CACHE_DIR='/fs/clip-scratch/nehasrik/paraphrase-nlu/cache'
python /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/roberta/defeasible/run_glue.py \
	--model_name_or_path microsoft/deberta-v3-large \
	--tokenizer_name microsoft/deberta-v3-large \
	--use_fast_tokenizer false \
	--cache_dir $CACHE_DIR \
	--do_train \
	--train_file /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/data_selection/defeasible/social/analysis_model_examples/train_examples.csv \
	--do_eval \
	--validation_file /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/data_selection/defeasible/social/analysis_model_examples/dev_examples.csv \
	--max_seq_length 128 \
	--per_device_train_batch_size=16 \
	--per_gpu_eval_batch_size=16 \
	--seed 42 \
	--save_strategy epoch \
	--learning_rate 5e-6 \
	--num_train_epochs 2 \
	--warmup_steps 50 \
	--overwrite_output_dir \
	--output_dir /fs/clip-scratch/nehasrik/paraphrase-nlu/deberta-models/d-social-deberta-v3-large

# TRAINING LARGE MODELS: lr=5e-6, seed=42, batch_size=16, epochs=2
# Base Models: lr=5e-6, seed=42, batch_size=64, epochs=2

#   --gradient_accumulation_steps 8 \
#	--do_lower_case \
# 	--output_attn \
#	--task_name $TASK_NAME \