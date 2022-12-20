#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test

dlrm_pt_bin="python dlrm_s_pytorch.py"

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
$dlrm_pt_bin \
    --arch-sparse-feature-size=11 \
    --arch-mlp-bot="3-512-256-64-11" \
    --arch-mlp-top="512-256-728" \
    --data-generation=hahow \
    --inputDir=./input/hahow \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=2 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --num-workers=16 \
    --use-gpu

echo "done"
