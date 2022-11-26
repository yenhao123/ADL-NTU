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


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    padValue = tag2idx['O']

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len,padValue)
    test_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,collate_fn=dataset.collate_fn)

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

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    # load weights into model
    model.eval()
    results = []
    for i,data in enumerate(test_loader):
        data["tokens"],ids,oriLen = data["tokens"].to(device),data["idx"],data["originalLen"]
        testPred,hidden = model(data)
        testPred = testPred.max(1)[1].cpu().data.numpy()
        res = [[dataset.idx2tag(p) for p in pred] for pred in testPred]
        #print(res)
        #exit(0)
        for j in range(len(res)):
            key = ids[j]
            value = " ".join(res[j][:oriLen[j]])
            row = {ids[j]:value}
            results.append(row)
        with open(args.pred_file,'w') as f:
            writer = csv.DictWriter(f,fieldnames=["id","tags"])
            writer.writeheader()
            for result in results:
                for key,value in result.items():
                    writer.writerow({"id":key,"tags":value})


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
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
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
