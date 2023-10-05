# ADL final README
###### tags: `台大`

learn

## Environment
```
# If you have conda, we recommend you to build a conda environment called "adl-hw3"
conda activate adl-fp-rechub
pip install torch-rechub


pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Dataset Prepare
```
# load hahow.zip
mkdir input
put hahow.zip and unzip it

# load train_summary.csv
sh download_dataset.sh
mv summary.csv input
```


### Training
```
sh train.sh
```

修改參數
```
input_dir : input 放置的目錄
```

### Inference
```
```
