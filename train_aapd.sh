#!/usr/bin/env bash
RUN_NAME=$1

seed=42
OUTPUT_DIR=models/$RUN_NAME
CACHE_DIR=.cache
TRAIN_FILE=./data/aapd/train_data.jsonl
NAME=AAPD

if [ -d $OUTPUT_DIR ]; then
  echo  "Output path: $OUTPUT_DIR already exists, please remove it first or set RUN_NAME "
  exit 0
fi

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data aapd --batch 80 --name $NAME \
  --update 1 --layer 4 --eval_step 1000 \
  --warmup 5 --tau 0.1 \
  --label_cpt ./data/aapd/aapd.taxonomy \
  --lr 1e-5 \
  --lamb 0.1 \
  --head 4 \
  --contrast 1 \
  --max_epoch 500 \
  --contrast_mode attentive \
  --wandb_name $NAME \
  --device cuda:0 \
  --accelerator gpu \
  --gpus "0" \
  --accumulate_step 1 \
  --softmax_entropy \
  --lamb_1 0.5 \
  --wandb