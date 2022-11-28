accelerate launch test.py \
    --model_name_or_path model/best_prefix20 \
    --test_file data/public.jsonl \
    --source_prefix "summarize: " \
    --preprocessing_num_workers 8 \
    --text_column "maintext" \
    --summary_column "title" \
    --outputPath results/eval/output.jsonl