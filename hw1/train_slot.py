import json
import pickle
from argparse import Namespace
from slot_arg_train import parse_args
from typing import Dict

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab,set_seed

import numpy as np
import time

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    set_seed(0)
    #load text mapping method
    with open(args.cache_dir / "vocab.pkl","rb") as f:
        vocab: Vocab = pickle.load(f)

    #load label mapping method
    tag2idxPath = args.cache_dir / "tag2idx.json"
    tags2idx: Dict[str,int] = json.loads(tag2idxPath.read_text())
    print(tags2idx)
    padValue = tags2idx['O']
    
    #load data
    dataPath = {split:args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split:json.loads(dataPath[split].read_text()) for split in SPLITS}
    #Dataset & Dataloader
    datasets = {split:SeqTaggingClsDataset(data[split],vocab,tags2idx,args.max_len,padValue) for split in SPLITS}  
    dataloaders = {"train":DataLoader(datasets["train"],args.batch_size,shuffle=False,collate_fn=datasets["train"].collate_fn),"eval":DataLoader(datasets["eval"],args.batch_size,shuffle=False,collate_fn=datasets["eval"].collate_fn)} 
    embeddings = torch.load(args.cache_dir /  "embeddings.pt")
    
    #model
    print("num_class:" + str(datasets["train"].get_num_class()))
    device = args.device
    model = SeqTagger(embeddings=embeddings,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          bidirectional=args.bidirectional,
                          max_len=args.max_len,
                          num_class=datasets["train"].get_num_class()).to(device)
    
    #optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model.parameters(),lr=args.lr)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.gamma)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2,eta_min=1e-9)
    loss = nn.CrossEntropyLoss()
    #train
    valLossMin = 2
    epochPbar = trange(args.num_epoch,desc="Epoch")
    startTime = time.time()
    for epoch in epochPbar:
        trainLoss,trainAcc,valLoss,valAcc = 0.0,0.0,0.0,0.0
         
        model.train()
        for i,data in enumerate(dataloaders['train']) :
            data["tokens"],data["tags"] = data["tokens"].to(device),data["tags"].to(device)
            optimizer.zero_grad()
            trainPred,_ = model(data)
            batchLoss = loss(trainPred,data["tags"])
            batchLoss.backward()
            optimizer.step()
            
            trainAcc += torch.all(data["tags"].eq(trainPred.max(1)[1]), dim=1).sum().item()
            trainLoss += batchLoss.item() * data["tokens"].shape[0]
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(dataloaders['eval']):
                data["tokens"],data["tags"] = data["tokens"].to(device),data["tags"].to(device)
                valPred,_ = model(data)
                batchLoss = loss(valPred,data["tags"])

                valAcc += torch.all(data["tags"].eq(valPred.max(1)[1]), dim=1).sum().item()
                valLoss += batchLoss.item() * data["tokens"].shape[0]
        
        trainLoss = trainLoss/datasets["train"].__len__()
        trainAcc = trainAcc/datasets["train"].__len__()
        valLoss = valLoss/datasets["eval"].__len__()
        valAcc = valAcc/datasets["eval"].__len__()
        print("[%03d/%03d] Train Loss: %3.6f Train Acc: %3.6f| Val Loss: %3.6f Val Acc: %3.6f" % (epoch,args.num_epoch,trainLoss,trainAcc,valLoss,valAcc))
        
        if valLoss <= valLossMin:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valLossMin,
                valLoss))
            torch.save(model.state_dict(), str(args.ckpt_dir) + "/" + str(args.outputName) + ".pt")
            valLossMin = valLoss
        
    endTime = time.time()
    print("time:" + str(endTime - startTime))

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
