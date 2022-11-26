#"${1}": path to the context file.
#"${2}": path to the testing file.
#"${3}": path to the output predictions.


accelerate launch multiple-choice/predict.py \
    --model_name_or_path tmp/bertMultipleChoice \
    --context_file "${1}" \
    --test_file "${2}" \
    --out_file  mutiChoiceRes.json \
    --pad_to_max_length
wait

accelerate launch question-answering/predict.py \
    --model_name_or_path tmp/roberta-largeBest \
    --test_file mutiChoiceRes.json \
    --context_file dataset/context.json \
    --output_path "${3}" \
    --output_dir ./ 
wait
echo "Done"
