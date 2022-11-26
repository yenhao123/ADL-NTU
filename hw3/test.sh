python3 test.py \
    --model_name_or_path model/epoch20 \
    --test_file data/sample_test.jsonl \
    --source_prefix "summarize: " \
    --preprocessing_num_workers 8 \
    --text_column "maintext" \
    --summary_column "title" 