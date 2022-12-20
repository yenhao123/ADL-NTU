accelerate launch train.py  \
    --model_name_or_path google/mt5-small \
    --train_file data/train.jsonl \
    --validation_file data/public.jsonl \
<<<<<<< HEAD
    --output_dir model/epoch10Batch4 \
=======
    --output_dir model/batch4 \
>>>>>>> 07cdf0187da457dc500e933f2b8f75e960a93d2d
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 10 \
    --source_prefix "summarize: " \
    --text_column "maintext" \
    --summary_column "title" \
    --with_tracking