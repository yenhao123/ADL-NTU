accelerate launch run_qa_no_trainer.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --train_file ../dataset/train.json \
  --validation_file ../dataset/valid.json \
  --context_file ../dataset/context.json \
  --learning_rate 3e-5 \
  --max_seq_length 512 \
  --checkpointing_steps 3000 \
  --output_dir ../tmp/roberta-large