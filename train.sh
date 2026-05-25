#!/bin/bash
CONFIG=${1:-configs/llada_code.yaml}

mkdir -p log
accelerate launch \
  trainer/main_rl.py config=$CONFIG 2>&1 | tee log/train.log
