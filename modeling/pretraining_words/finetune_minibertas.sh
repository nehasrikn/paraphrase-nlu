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
export CACHE_DIR='/fs/clip-projects/rlab/nehasrik/cache'

ROOT_DIR='/fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/pretraining_words/minibertas/'

for item in snli atomic social;
do
    python /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/roberta/defeasible/run_glue.py \
    	--model_name_or_path nyu-mll/roberta-base-100M-2 \
    	--tokenizer_name nyu-mll/roberta-base-100M-2 \
    	--use_fast_tokenizer false \
    	--cache_dir $CACHE_DIR \
    	--do_train \
    	--train_file /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/data_selection/defeasible/$item/analysis_model_examples/train_examples.csv \
    	--do_eval \
    	--validation_file /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/data_selection/defeasible/$item/analysis_model_examples/dev_examples.csv \
    	--max_seq_length 128 \
    	--per_device_train_batch_size=64 \
    	--per_gpu_eval_batch_size=64 \
    	--seed 42 \
    	--save_strategy epoch \
    	--learning_rate 5e-6 \
    	--num_train_epochs 2 \
    	--overwrite_output_dir \
    	--output_dir $ROOT_DIR/100M-pretraining-words/d-$item-roberta-base-100M-2

done

# TRAINING LARGE MODELS: lr=5e-6, seed=42, batch_size=16, epochs=2
# Base Models: lr=5e-6, seed=42, batch_size=64, epochs=2
# MiniBERTas:
	# 10M-2: tried 5e-6, batch_size=64, epochs=2

#   --gradient_accumulation_steps 8 \
#	--do_lower_case \
# 	--output_attn \
#	--task_name $TASK_NAME \