#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --nodelist=clip08

module load cuda
source /fs/clip-projects/rlab/nehasrik/init_conda.sh
conda activate para-nlu

cd /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/roberta/defeasible/
export TASK_NAME="mrpc"
export CACHE_DIR='/fs/clip-scratch/nehasrik/paraphrase-nlu/cache'

sizes=("0.01" "0.05" "0.1" "0.5")

split='social'

experiment_dir="/fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/finetuning_volume"


for size in "${sizes[@]}"; do
  trainfile="train_examples_${size}.csv"
  echo "Processing $trainfile"
  # You can perform operations on the file with "$filename" here
  python run_glue.py \
	--model_name_or_path roberta-base \
	--tokenizer_name roberta-base \
	--use_fast_tokenizer false \
	--cache_dir $CACHE_DIR \
	--do_train \
	--train_file /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/experiments/pretraining-vs-finetuning/finetuning_data/$split/$trainfile \
	--do_eval \
	--validation_file /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/data_selection/defeasible/$split/analysis_model_examples/dev_examples.csv \
	--max_seq_length 128 \
	--per_device_train_batch_size=64 \
	--per_gpu_eval_batch_size=64 \
	--seed 42 \
	--save_strategy epoch \
	--learning_rate 5e-6 \
	--num_train_epochs 2 \
	--overwrite_output_dir \
	--output_dir $experiment_dir/chkpts/$split/d-$split-roberta-base-$size
done




