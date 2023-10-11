from transformers import AutoTokenizer
from fairseq.data import data_utils
import torch
from tqdm import tqdm
import argparse
import os
from model.eval import evaluate

from model import MInterface
from data import DInterface
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import manifold

import utils
import pandas as pd
import numpy as np
import json
from model.text_attention import generate
from utils import get_hierarchy_info, save_results
import pickle
torch.set_float32_matmul_precision("high")

class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='wos', choices=['wos', 'nyt', 'rcv1', 'bgc', 'patent', 'aapd'], help='Dataset.')
parser.add_argument('--label_cpt', type=str, default='data/nyt/nyt.taxonomy', help='Label hierarchy file.')
parser.add_argument('--batch', type=int, default=12, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=6, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=2000, type=int, help='Warmup steps.')
parser.add_argument('--contrast', default=1, type=int, help='Whether use contrastive model.')
parser.add_argument('--contrast_mode', default='attentive', type=str, choices=['label_aware', 'fusion', 'attentive', 'simple_contrastive', 'straight_through'], help='Contrastive model type.')
parser.add_argument('--graph', default=1, type=int, help='Whether use graph encoder.')
parser.add_argument('--layer', default=1, type=int, help='Layer of Graphormer.')
parser.add_argument('--multi', default=True, action='store_false', help='Whether the task is multi-label classification.')
parser.add_argument('--lamb', default=1, type=float, help='lambda')
parser.add_argument('--thre', default=0.02, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=3, type=int, help='Random seed.')
parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging.')
parser.add_argument('--tf_board', default=False, action='store_true', help='Use tensorboard for logging.')
parser.add_argument('--eval_step', default=1000, type=int, help='Evaluation step.')
parser.add_argument('--head', default=4, type=int, help='Number of heads.')
parser.add_argument('--max_epoch', default=100, type=int, help='Maximum epoch.')
parser.add_argument('--wandb_name', default='supContrastiveHMTC', type=str, help='Wandb project name.')
parser.add_argument('--test', default=False, action='store_true', help='Test mode.')
parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint path.')
parser.add_argument('--accelerator', default='ddp', type=str, help='Accelerator for training.')
parser.add_argument('--gpus', default='0', type=str, help='GPU for training.')
parser.add_argument('--test_only', default=False, action='store_true', help='Test only mode.')
parser.add_argument('--test_checkpoint', default=None, type=str, help='Test checkpoint path.')
parser.add_argument('--accumulate_step', default=1, type=int, help='Gradient accumulate step.')
parser.add_argument('--decay_epochs', default=0, type=int, help='Decay epochs.')
parser.add_argument('--softmax_entropy', default=False, action='store_true', help='Use softmax+entropy loss.')
parser.add_argument('--use_decoder', default=False, action='store_true', help='Use decoder.')
parser.add_argument('--ignore_contrastive', default=False, action='store_true', help='Ignore contrastive loss.')
parser.add_argument('--count_weight', default=0, type=float, help='Weight for count loss.')
parser.add_argument('--do_simple_label_contrastive', default=False, action='store_true', help='Use simple label contrastive loss.')
parser.add_argument('--do_weighted_label_contrastive', default=False, action='store_true', help='Use simple label contrastive loss v2.')
parser.add_argument('--lamb_1', default=0.0, type=float, help='Weight for simple label contrastive loss.')
parser.add_argument('--add_path_reg', default=False, action='store_true', help='Add path regularization.')
parser.add_argument('--path_reg_weight', default=0.0, type=float, help='Weight for path regularization.')
parser.add_argument('--path_reg_weight_adjusted', default=False, action='store_true', help='Weight for path regularization.')
parser.add_argument('--ignore_path_reg', default=False, action='store_true', help='Weight for path regularization.')
parser.add_argument('--hamming_dist_mode', default=None, type=str, help='Hamming distance mode.')

def get_root(path_dict, n):
    ret = []
    while path_dict[n] != n:
        ret.append(n)
        n = path_dict[n]
    ret.append(n)
    return ret


if __name__ == '__main__':
    args = parser.parse_args()
    device = args.device
    print(args)
    loggers = []
    wandb_logger = None
    if args.wandb:
        # import wandb
        # wandb.init(config=args, project='supContrastiveHMTC', name=args.wandb_name)
        wandb_logger = WandbLogger(name=args.wandb_name, project='supContrastiveHMTC')
        loggers.append(wandb_logger)

    # log the args to wandb
    if args.wandb:
        wandb_logger.log_hyperparams(args)
    utils.seed_torch(args.seed)
    pl.seed_everything(args.seed)
        
    args.name = args.data + '-' + args.name
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_path = os.path.join('data', args.data)
    # This load the pertrained bert model
    # the following code needs to have label_dict, num_class, hiera, r_hiera, label_depth, new_label_dict
    if args.data == 'nyt':
        # label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
        # label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    
    # load the new label dict
    
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            new_label_dict = pickle.load(f)

        label_dict = new_label_dict
        # new_label_dict = label_dict

        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'nyt.taxonomy'))
        depths = [len(l.split('/')) - 1 for l in new_label_dict.values()]

        num_class = len(label_dict)

    elif args.data == 'rcv1':
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'rcv1.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)

        new_label_dict = label_dict

        r_hiera = {new_label_dict[_label_dict[k]]: v if (v == 'Root') else new_label_dict[_label_dict[v]] for k, v in r_hiera.items()}

        # {label_name: label_id}
        label_dict = {v: k for k, v in _label_dict.items()}
        num_class = len(label_dict)
        
        # rcv_label_map = pd.read_csv(os.path.join(data_path, 'rcv1_v2_topics_desc.csv'))
        # rcv_label_amp = dict(zip(rcv_label_map['topic_code'], rcv_label_map['topic_name']))

        # new_label_dict = {k: rcv_label_amp[v] for k, v in label_dict.items()}
        depths = [label_depth[name] for id, name in label_dict.items()]

    elif args.data == 'bgc':
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'bgc.taxonomy'))
        label_dict = {v: k for k, v in _label_dict.items()}
        new_label_dict = label_dict
        num_class = len(label_dict)
        depths = list(label_depth.values())
    elif args.data == 'patent':
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'patent.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)
        label_dict = {v: k for k, v in _label_dict.items()}
        new_label_dict = label_dict

        num_class = len(label_dict)
        depths = list(label_depth.values())
    elif args.data == 'aapd':
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'aapd.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)
        label_dict = {v: k for k, v in label_dict.items()}
        new_label_dict = label_dict
        num_class = len(label_dict)
        depths = [label_depth[name] for id, name in label_dict.items()]
    elif args.data == 'wos':
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(os.path.join(data_path, 'wos.taxonomy'))
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)
        label_dict = {v: k for k, v in label_dict.items()}
        new_label_dict = label_dict
        num_class = len(label_dict)
        depths = [label_depth[name] for id, name in label_dict.items()]
        
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.makedirs(os.path.join('checkpoints', args.name))
    # store the label_dict as a json file
    with open(os.path.join('checkpoints', args.name, 'label_dict.json'), 'w') as f:
        json.dump(label_dict, f)

    def get_path(label):
        path = []
        
        if args.data == 'rcv1':
            _, _, rcv_r_hiera, _ = get_hierarchy_info(os.path.join(data_path, 'rcv1.taxonomy'))
            while label != 'Root':
                path.insert(0, label)
                label = rcv_r_hiera[label]
        # label_name = label_dict[label]
        else:
            while label != 'Root':
                path.insert(0, label)
                label = r_hiera[label]
        return path
    
    if ('nyt' in data_path) or ('aapd' in data_path) or ('wos' in data_path):
        print(label_dict)
        label_path = {v: get_path(v) for k, v in label_dict.items()}
    elif ('rcv' in data_path):
        print(_label_dict)
        label_path = {k: get_path(k) for k, v in _label_dict.items()}
    elif ('bgc' in data_path):
        label_path = {k: get_path(k) for k, v in _label_dict.items()}
    else:
        label_path = {k: get_path(k) for k, v in label_dict.items()}

    args.label_path = label_path

    checkpoint_model_path = os.path.join('checkpoints', args.test_checkpoint)
    args.save_path = checkpoint_model_path + '_save/'
    args.skip_batch_sampling = False

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model = MInterface.load_from_checkpoint(checkpoint_model_path)
    data_module = DInterface(args=args, tokenizer=tokenizer, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict)

    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    model.model = model.model.to(device)

    features = []
    labels = []
    # random sample from test_loader
    for i, batch in enumerate(test_loader):
        input_ids, label, idx = batch
        padding_mask = input_ids != tokenizer.pad_token_id
        
        # check the input_ids device
        print(input_ids.device)

        outputs = model.model(input_ids, padding_mask, labels=label, return_dict=True)

        print(outputs.keys())

        num_class = outputs['features'].shape[1]
        feature = outputs['features'].cpu().detach() # batch, num_class, hidden_size
        # flat the first two dimensions
        feature = feature.view(-1, feature.shape[-1]) # batch * num_class, hidden_size
        label = label.cpu().detach() # batch, num_class
        
        label = label.view(-1) # batch * num_class

        # select those index in feature that label is larger than 0
        feature = feature[label > 0]
        label = torch.nonzero(label > 0, as_tuple=True)[0]
        label = label % num_class

        features.append(feature)
        labels.append(label)

        print(label.shape)
        if i >= 3:
            break

    features_tensor = torch.cat(features, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    tsne = manifold.TSNE(n_components=2, random_state=0)
    y = tsne.fit_transform(features_tensor)

    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))
    value2slot = {}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = i
            
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib

    colormap = matplotlib.cm.Set1.colors
    color = []
    inv_label_dict = {v: k for k, v in label_dict.items()}
    father2idx = {v: i for i, v in enumerate(np.where(np.array(depths) == 1)[0].tolist())}

    def get_father(x):
        if value2slot[x] == x:
            return x
        else:
            return get_father(value2slot[x])

    def get_upper_father(x):
        label = label_dict[x]
        while label != 'Root':
            label_idx = inv_label_dict[label]
            label = r_hiera[label]
        return father2idx[label_idx]

    labels_np = labels_tensor.cpu().numpy()
    for i in range(len(labels_tensor)):
        color.append(colormap[get_upper_father(labels_np[i])])
    plt.scatter(y[:, 0], y[:, 1], c=color)
    plt.axis('off')

    plt.savefig(os.path.join(args.save_path, 'tsne.png'), bbox_inches='tight', dpi=300)