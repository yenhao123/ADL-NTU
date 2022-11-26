import numpy
import json
import csv
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

def main(args):
    #open eval file
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    print(tag2idx)
    padValue = tag2idx['O']
    print(padValue)

    data = json.loads(Path("data/slot/eval.json").read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len, padValue)
    valLoader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,collate_fn=dataset.collate_fn)

    #train
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = args.device
    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        args.max_len,
        dataset.get_num_class(),
    ).to(device)

    ckpt = torch.load("ckpt/slot/best.pt")
    model.load_state_dict(ckpt)
    model.eval()
    predResults,predLabels = None,None
    for i,data in enumerate(valLoader):
        data["tokens"],tags,ids,oriLen = data["tokens"].to(device),data["tags"],data["idx"],data["originalLen"]
        testPred,hidden = model(data)
        testPred = testPred.max(1)[1].cpu().data.numpy()
        pred = [[dataset.idx2tag(p) for p in pred] for pred in testPred]
        label = [[dataset.idx2tag(t) for t in tag] for tag in tags.numpy()]
        if predResults == None:
            predResults = pred
            predLabels = label
        else:
            predResults += pred
            predLabels += label
        #print(predResults)
        
    c = classification_report(predResults,predLabels,mode='strict',scheme=IOB2)
    print(c)
    exit(0)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:1"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
