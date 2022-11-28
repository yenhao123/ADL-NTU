accelerate launch eval.py \
    --model_name_or_path model/best_prefix20 \
    --validation_file data/public.jsonl \
    --preprocessing_num_workers 8 \
    --source_prefix "summarize: " \
    --text_column "maintext" \
    --summary_column "title" \
    --outputPath results/eval/output.jsonl \
    --lossANDmetricPath lossANDmetric/eval/result.json 