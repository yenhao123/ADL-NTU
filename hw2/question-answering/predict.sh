accelerate launch predict.py \
    --model_name_or_path ../tmp/roberta-large \
    --test_file ../results/multipleChoiceRes.json \
    --context_file ../dataset/context.json \
    --output_path ../results/robertaQA.csv \
    --output_dir ./ 