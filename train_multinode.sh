#!/bin/bash
CONFIG=${1:-configs/llada_code.yaml}

mkdir -p log
accelerate launch \
  --num_machines=$WORLD_SIZE \
  --machine_rank=$RANK \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --num_processes=$((WORLD_SIZE * 8)) \
  trainer/main_rl.py config=$CONFIG 2>&1 | tee log/train.log
