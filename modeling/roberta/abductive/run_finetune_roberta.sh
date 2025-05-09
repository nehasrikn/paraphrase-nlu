module load cuda

python run_multiple_choice.py \
--model_name_or_path roberta-large \
--do_train \
--do_eval \
--learning_rate 5e-6 \
--num_train_epochs 2 \
--use_fast_tokenizer false \
--output_dir chkpts/roberta-large-anli \
--per_gpu_eval_batch_size=8 \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--save_strategy epoch \
--seed 42 \
--cache_dir '/fs/clip-scratch/nehasrik/paraphrase-nlu/paraphrase-nlu/modeling/hf-cache' \
