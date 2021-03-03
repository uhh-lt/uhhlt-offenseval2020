python run_language_modeling.py \
  --train_data_file=datasets/OffensEval20/training_filtered.txt \
  --eval_data_file=dataset/OffensEval20/validation.txt \
  --per_gpu_train_batch_size=3 \
  --output_dir=models \
  --save_total_limit=5 \
  --cache_dir=cache \
  --overwrite_output_dir \
  --overwrite_cache \
  --model_type=roberta \
  --model_name_or_path=roberta-large \
  --line_by_line \
  --mlm \
  --do_train \
  --num_train_epochs=1 \
  --learning_rate=2e-5 \
  --seed=9721