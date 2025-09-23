#!/bin/bash
checkpoint=checkpoint1epoch
ds=high_app_split # one of "low_app_split", "low_device_split", "low_random_split", "low_task_split" "high_app_split", "high_device_split", "high_random_split", "high_task_split"
DIR=`pwd`

exp_name=OdysseyAgent_$ds
mkdir -p output/"$exp_name"

GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo $ds
echo $checkpoint
torchrun --nproc_per_node $GPUS_PER_NODE ../eval_mm/evaluate_GUIOdyssey.py \
    --checkpoint $checkpoint --dataset $ds --batch-size 28 --his_len 4