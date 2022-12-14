accelerate launch test.py \
    --model_name_or_path best_prefix20 \
    --test_file data/sample_test.jsonl \
    --source_prefix "summarize: " \
    --preprocessing_num_workers 8 \
    --text_column "maintext" \
    --outputPath results/eval/output.jsonl