# Sample Code for Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection - Training
```shell
python train_intent.py
```
## Slot Tagging - Training
```shell
python train_slot.py
```

## Testing
```shell
bash download.sh
bash ./intent_cls.sh data/intent/test.json intent.pred.csv
bash ./slot_tag.sh data/slot/test.json slot.pred.csv
```
