# ADL hw3 README
###### tags: `台大`

learn
* Use the hot platform - Hugging Face
* Understand the summarization model - mT5

## Environment
```
# If you have conda, we recommend you to build a conda environment called "adl-hw3"
conda activate adl-hw3
conda install python=3.8
```


setup tw_rogue
```
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
```
>when data can not download
```
1. download the data by TA
2. unzip data.zip to data
3. put data to .cache/ckiptagger
```


setup transformers
```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
>Then cd in the example folder of your choice and run
```
pip install -r requirements.txt
```

setup env
```
make
pip install -r requirements.txt
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorflow
```

### Training
```
sh train.sh
```

修改參數
```
train file : train data path
validation_file : validation data path
output_dir : model path
```

### Evaluation
```
sh eval.sh
```

修改參數
```
model_name_or_path : put model path
validation_file : validation data path
```

### Testing
```
sh download.sh
sh run.sh ${1} ${2} //${1}: path to the input file ; {2}: path to the output file
```
