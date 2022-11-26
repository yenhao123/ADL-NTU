import json
import pickle
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import numpy as np

from dataset import SeqClsDataset
from torch.utils.data import DataLoader
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,collate_fn=dataset.collate_fn)
    #for data in test_loader:
    #    print(data)
    #exit(0)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )

    device = args.device
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    # load weights into model
    model.eval()
    results = []
    for i,data in enumerate(test_loader):
        test_data,ids = data["text"].to(device),data["id"]
        test_pred,hidden = model(test_data)
        test_pred = torch.max(test_pred.cpu().data,1)[1].numpy()
        res = []
        for pred in test_pred:
            res.append(dataset.idx2label(pred))
        
        for j in range(len(res)):
            key = ids[j]
            value = res[j]
            row = {ids[j]:value}
            results.append(row)
    #print(results)
    #exit(0)    
    with open(args.pred_file,'w') as f:
            writer = csv.DictWriter(f,fieldnames=["id","intent"])
            writer.writeheader()
            for result in results:
                for key,value in result.items():
                    writer.writerow({"id":key,"intent":value})

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
