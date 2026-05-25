#!/bin/bash
CONFIG=${1:-configs/eval/eval_llada_code.yaml}

mkdir -p log
accelerate launch \
  trainer/eval.py config=$CONFIG 2>&1 | tee log/eval.log
