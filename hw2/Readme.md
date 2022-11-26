# Homework 2 ADL NTU

###### tags: `台大`

learn
* Use the hot platform - Hugging Face
* Know the implementaion of BERT

## Environment
```
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw2
pip install -r requirements.txt
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Training
```
cd multiple-choice
sh run_no_trainer.sh
sh predict.sh
sh run.sh
sh predict.sh
```

結果
>results/robertaQA.csv

### Testing
```
sh download.sh
sh run.sh dataset/context.json dataset/test.json prediction.csv
```