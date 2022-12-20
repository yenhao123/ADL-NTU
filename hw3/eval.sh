accelerate launch eval.py \
<<<<<<< HEAD
    --model_name_or_path model/prefix20 \
=======
    --model_name_or_path model/best_prefix20 \
>>>>>>> 07cdf0187da457dc500e933f2b8f75e960a93d2d
    --validation_file data/public.jsonl \
    --preprocessing_num_workers 8 \
    --source_prefix "summarize: " \
    --text_column "maintext" \
<<<<<<< HEAD
    --summary_column "title" 
=======
    --summary_column "title" \
    --outputPath results/eval/output.jsonl \
    --lossANDmetricPath lossANDmetric/eval/result.json 
>>>>>>> 07cdf0187da457dc500e933f2b8f75e960a93d2d
