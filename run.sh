#!/bin/bash
# set run
echo "program first start..."
echo "please check and update the dataset value"
echo "model will save in models, logger will save in logs"
python train.py --dataset "WebKB" --epochs 100 --iteration 5
