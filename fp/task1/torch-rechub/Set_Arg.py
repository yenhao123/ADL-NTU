import pandas as pd
import os
import datetime
import numpy as np
import json
import argparse

def set_arg():
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    parser.add_argument("--input_dir", type=str,default="./input/hahow")
    parser.add_argument("--load_cache", action="store_true")
    # model
    parser.add_argument('--model_name', default='dssm')
    parser.add_argument('--epoch', type=int, default=10)  #5
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4096)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')  #cuda:0
    parser.add_argument('--save_dir', default='./output/')
    parser.add_argument('--seed', type=int, default=2022)

    
    args = parser.parse_args()
    return args