#${1}: path to the input file
#${2}: path to the output file


accelerate launch test.py \
    --model_name_or_path best_prefix20 \
    --test_file "${1}" \
    --source_prefix "summarize: " \
    --preprocessing_num_workers 8 \
    --text_column "maintext" \
    --outputPath "${2}"

wait
echo "Done"