import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument(
        "--outputName",
        type=str,
        help="Name to save the model file.",
        default="best.pt",
    )
    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.9)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)
    
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:1"
    )
    args = parser.parse_args()
 
    return args

