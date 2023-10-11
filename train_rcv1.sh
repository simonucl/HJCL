#!/usr/bin/env bash
RUN_NAME=$1

seed=42
OUTPUT_DIR=models/$RUN_NAME
CACHE_DIR=.cache
TRAIN_FILE=./data/rcv1/rcv1_train_all_generated_tl.json
NAME=RCV_1

if [ -d $OUTPUT_DIR ]; then
  echo  "Output path: $OUTPUT_DIR already exists, please remove it first or set RUN_NAME "
  exit 0
fi

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data rcv1 --batch 75 --name $NAME \
  --update 1 --layer 4 --eval_step 1000 \
  --warmup 5 --tau 0.1 \
  --label_cpt ./data/rcv1/rcv1.taxonomy \
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
  --lamb_1 0.2 \
  --wandb
