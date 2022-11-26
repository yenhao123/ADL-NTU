accelerate launch predict.py \
    --test_file ../dataset/test.json \
    --context_file ../dataset/context.json \
    --model_name_or_path ../tmp/bertMultipleChoice \
    --out_file  ../results/mutiChoiceRes.json \
    --pad_to_max_length