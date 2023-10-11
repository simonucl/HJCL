#!/usr/bin/env bash
RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
  RUN_NAME=nyt
fi

seed=42
OUTPUT_DIR=models/$RUN_NAME
CACHE_DIR=.cache
TRAIN_FILE=./data/nyt/nyt_train_all_generated_tl.json
NAME=sample_attentive_wo_batch_sampling
batch=80
epoch=200

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data nyt --batch 75 --name $NAME \
  --update 1 --layer 4 --eval_step 1000 \
  --warmup 5 --tau 0.1 \
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
  --do_weighted_label_contrastive \
  --lamb_1 0.5 \
  --wandb