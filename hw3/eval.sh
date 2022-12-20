accelerate launch eval.py \
    --model_name_or_path model/prefix20 \
    --validation_file data/public.jsonl \
    --preprocessing_num_workers 8 \
    --source_prefix "summarize: " \
    --text_column "maintext" \
    --summary_column "title" 