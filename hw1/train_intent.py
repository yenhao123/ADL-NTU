import arg
import json
import pickle
import time
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    #the list of the text mapping
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    #the list of the intent mapping
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    #使用SeqClsDataset來包裝data
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    train_loader = DataLoader(datasets["train"],batch_size=args.batch_size,shuffle=True,collate_fn=datasets["train"].collate_fn)
    dev_loader = DataLoader(datasets["eval"],batch_size=args.batch_size,shuffle=False,collate_fn=datasets["eval"].collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    #training
    num_class = 150
    device = args.device
    model = SeqClassifier(embeddings=embeddings,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          bidirectional=args.bidirectional,
                          num_class=num_class).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    epoch_pbar = trange(args.num_epoch,desc="Epoch")
    start_time = time.time()
    valLossMin = 2
    for epoch in epoch_pbar:
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        
        model.train()
        for i,data in enumerate(train_loader):
            train_data,label = data["text"].to(device),data['intent'].to(device)
            optimizer.zero_grad()
            train_pred,hidden = model(train_data)
            batch_loss = loss(train_pred,label)
            batch_loss.backward()
            optimizer.step()
            train_acc += torch.sum(torch.max(train_pred.cpu().data,1)[1] == data["intent"])
            train_loss += batch_loss.item()
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(dev_loader):
                val_pred,val_hidden = model(data["text"].to(device))
                batch_loss = loss(val_pred,data["intent"].to(device))

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(),axis=1) == data["intent"].numpy())
                val_loss += batch_loss.item()
        train_loss = train_loss/datasets["train"].__len__()
        train_acc = train_acc/datasets["train"].__len__()
        val_loss = val_loss/datasets["eval"].__len__()
        val_acc = val_acc/datasets["eval"].__len__()
        print("[%03d/%03d] Train Loss: %3.6f Train Acc: %3.6f| Val Loss: %3.6f Val Acc: %3.6f" % (epoch,args.num_epoch,train_loss,train_acc,val_loss,val_acc))
    
        if val_loss <= valLossMin:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valLossMin,
                val_loss))
            torch.save(model.state_dict(), str(args.ckpt_dir) + '/best.pt')
            valLossMin = val_loss
    
    endTime = time.time()
    print("time:" + str(endTime - startTime))    

if __name__ == "__main__":
    args = arg.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
