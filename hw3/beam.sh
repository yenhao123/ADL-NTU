accelerate launch generate.py \
    --model_name_or_path model/best_prefix20 \
    --validation_file data/public.jsonl \
    --preprocessing_num_workers 8 \
    --source_prefix "summarize: " \
    --text_column "maintext" \
    --summary_column "title" \
    --outputPath results/generation/beam.jsonl \
    --lossANDmetricPath lossANDmetric/generation/beam.json \
    --num_beams 5