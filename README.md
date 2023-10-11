# Instances and Labels: Hierarchy-aware Joint Supervised Contrastive Learning for Hierarchical Multi-Label Text Classification

This repository implements a joint contrastive learning objective for hierarchical text classification. This work has been submitted and being peer-review in EMNLP 2023 (Long Paper).

### Set up environments
```
conda env create -f environment.yaml
conda activate hjcl
```

## Preprocess

Please download the original dataset and then use these scripts.

### NYT

The original dataset can be acquired [here](https://catalog.ldc.upenn.edu/LDC2008T19).

```shell
cd ./data/nyt
python data_nyt.py
```

### RCV1-V2

The preprocess code could refer to the [repository of reuters_loader](https://github.com/ductri/reuters_loader) and we provide a copy here. The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement.

```shell
cd ./data/rcv1
python preprocess_rcv1.py
python data_rcv1.py
```

### BGC & AAPD

No preprocessing needed. Data is already provided under `data/bgc` and `data/aapd`, respectively.

## Train and testing
The scripts are already ready in bash file, which will automatically log the testing output to wandb. To train the model for datasets:

```
bash train_[DATASET_NAME].sh
```

Otherwise, if researcher would like to customize the training, they could pass in the following arguments:
```
usage: train.py [-h] [--lr LR] [--data {wos,nyt,rcv1,bgc,patent,aapd}]
                [--label_cpt LABEL_CPT] [--batch BATCH]
                [--early-stop EARLY_STOP] [--device DEVICE] --name NAME
                [--update UPDATE] [--warmup WARMUP] [--contrast CONTRAST]
                [--contrast_mode {label_aware,fusion,attentive,simple_contrastive,straight_through}]
                [--graph GRAPH] [--layer LAYER] [--multi] [--lamb LAMB]
                [--thre THRE] [--tau TAU] [--seed SEED] [--wandb] [--tf_board]
                [--eval_step EVAL_STEP] [--head HEAD] [--max_epoch MAX_EPOCH]
                [--wandb_name WANDB_NAME] [--checkpoint CHECKPOINT]
                [--accelerator ACCELERATOR] [--gpus GPUS] [--test_only]
                [--test_checkpoint TEST_CHECKPOINT]
                [--accumulate_step ACCUMULATE_STEP]
                [--decay_epochs DECAY_EPOCHS] [--softmax_entropy]
                [--ignore_contrastive] [--lamb_1 LAMB_1]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate.
  --data {wos,nyt,rcv1,bgc,patent,aapd}
                        Dataset.
  --label_cpt LABEL_CPT
                        Label hierarchy file.
  --batch BATCH         Batch size.
  --early-stop EARLY_STOP
                        Epoch before early stop.
  --device DEVICE
  --name NAME           A name for different runs.
  --update UPDATE       Gradient accumulate steps
  --warmup WARMUP       Warmup steps.
  --contrast CONTRAST   Whether use contrastive model.
  --contrast_mode {label_aware,fusion,attentive,simple_contrastive,straight_through}
                        Contrastive model type.
  --graph GRAPH         Whether use graph encoder.
  --layer LAYER         Layer of Graphormer.
  --multi               Whether the task is multi-label classification.
  --lamb LAMB           lambda
  --thre THRE           Threshold for keeping tokens. Denote as gamma in the
                        paper.
  --tau TAU             Temperature for contrastive model.
  --seed SEED           Random seed.
  --wandb               Use wandb for logging.
  --tf_board            Use tensorboard for logging.
  --eval_step EVAL_STEP
                        Evaluation step.
  --head HEAD           Number of heads.
  --max_epoch MAX_EPOCH
                        Maximum epoch.
  --wandb_name WANDB_NAME
                        Wandb project name.
  --checkpoint CHECKPOINT
                        Checkpoint path.
  --accelerator ACCELERATOR
                        Accelerator for training.
  --gpus GPUS           GPU for training.
  --test_only           Test only mode.
  --test_checkpoint TEST_CHECKPOINT
                        Test checkpoint path.
  --accumulate_step ACCUMULATE_STEP
                        Gradient accumulate step.
  --decay_epochs DECAY_EPOCHS
                        Decay epochs.
  --softmax_entropy     Use softmax+entropy loss.
  --ignore_contrastive  Ignore contrastive loss.
  --lamb_1 LAMB_1       Weight for weighted label contrastive loss.
```

## Citation
TBD