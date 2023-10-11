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
parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint path.')
parser.add_argument('--accelerator', default='ddp', type=str, help='Accelerator for training.')
parser.add_argument('--gpus', default='0', type=str, help='GPU for training.')
parser.add_argument('--test_only', default=False, action='store_true', help='Test only mode.')
parser.add_argument('--test_checkpoint', default=None, type=str, help='Test checkpoint path.')
parser.add_argument('--accumulate_step', default=1, type=int, help='Gradient accumulate step.')
parser.add_argument('--decay_epochs', default=0, type=int, help='Decay epochs.')
parser.add_argument('--softmax_entropy', default=False, action='store_true', help='Use softmax+entropy loss.')
parser.add_argument('--ignore_contrastive', default=False, action='store_true', help='Ignore contrastive loss.')
parser.add_argument('--lamb_1', default=0.0, type=float, help='Weight for weighted label contrastive loss.')

def get_root(path_dict, n):
    ret = []
    while path_dict[n] != n:
        ret.append(n)
        n = path_dict[n]
    ret.append(n)
    return ret


if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        import sys
        sys.exit(1)

    args.do_weighted_label_contrastive = True
    args.skip_batch_sampling = False
    args.hamming_dist_mode = 'depth_weight'


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

    args.depths = depths
    args.label_path = label_path

    # create depths by the number of '/' in the new label dict
    # depths = [len(l.split('/')) - 1 for l in new_label_dict.values()]

    if args.test_only:
        if args.test_checkpoint is None:
            raise ValueError('Please specify the checkpoint path for testing.')

        checkpoint_model_path = os.path.join('checkpoints', args.test_checkpoint)
        args.save_path = checkpoint_model_path + '_save/'
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if os.path.exists(os.path.join(args.save_path, 'attns.pkl')):
            # delete the attention files
            os.remove(os.path.join(args.save_path, 'attns.pkl'))
        if os.path.exists(os.path.join(args.save_path, 'indices.pkl')):
            # delete the indices files
            pass
        if os.path.exists(os.path.join(args.save_path, 'labels.pkl')):
            # delete the labels files
            os.remove(os.path.join(args.save_path, 'labels.pkl'))
        if os.path.exists(os.path.join(args.save_path, 'input_ids.pkl')):
            # delete the input_ids files
            os.remove(os.path.join(args.save_path, 'input_ids.pkl'))

        data_module = DInterface(args=args, tokenizer=tokenizer, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict)
        model = MInterface.load_from_checkpoint(checkpoint_model_path, args=args, num_labels=num_class, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict,
                                                new_label_dict=new_label_dict, r_hiera=r_hiera)

        trainer = Trainer(accelerator=args.accelerator, strategy='auto')
        trainer.test(model, datamodule=data_module)

        import sys
        sys.exit(1)

    data_module = DInterface(args=args, tokenizer=tokenizer, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict)


    if args.test_checkpoint is None:
        model = MInterface(args, num_labels=num_class, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict, new_label_dict=new_label_dict, r_hiera=r_hiera)
        args.save_path = os.path.join('checkpoints', args.name + '_save/')
        
    else:
        checkpoint_model_path = os.path.join('checkpoints', args.test_checkpoint)
        args.save_path = checkpoint_model_path + '_save/'


        model = MInterface.load_from_checkpoint(checkpoint_model_path, args=args, num_labels=num_class, label_depths=depths, device=device, data_path=data_path, label_dict=label_dict,
                                                new_label_dict=new_label_dict, r_hiera=r_hiera)
    # This load the contrastive and classification model
    # model = ContrastModel.from_pretrained('bert-base-uncased', args.batch, num_labels=num_class,
    #                                       contrast_loss=args.contrast, contrast_mode= args.contrast_mode,
    #                                       graph=args.graph, label_depths=depths, device=device,
    #                                       layer=args.layer, data_path=data_path, multi_label=args.multi,
    #                                       lamb=args.lamb, threshold=args.thre, tau=args.tau, head=args.head,
    #                                       label_cpt=args.label_cpt)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    if args.wandb:
        wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor='test/macro_f1', mode='max', save_top_k=1, 
                                          dirpath=os.path.join('checkpoints', args.name), filename= args.name + '-{epoch}-{val_loss:.2f}', save_on_train_epoch_end=False)
    trainer = Trainer(max_epochs=args.max_epoch, strategy="auto",
                      accelerator=args.accelerator, logger=wandb_logger if args.wandb else None,
                      accumulate_grad_batches=args.accumulate_step, default_root_dir=os.path.join('checkpoints', args.name),
                        gradient_clip_val=1.0, callbacks=[EarlyStopping(monitor='test/macro_f1', patience=5, mode='max', check_on_train_epoch_end=False), checkpoint_callback]
                        )
    trainer.fit(model, data_module)

    import sys
    sys.exit(1)

    # TODO Add evaluation code here

    # dev_raw_data = [tokenizer.decode(i['data'], skip_special_tokens=True) for i in dev]

    if args.test:
        if not args.checkpoint:
            raise ValueError('Please specify the checkpoint path.')
        model.load_state_dict(torch.load(args.checkpoint)['param'])
        model.to(device)
        model.eval()
        test_ct = 0
        test_loss = 0
        pbar = tqdm(test)
        # eval_data = Subset(dataset, split['train'][:1000])
        # dev_loader = DataLoader(eval_data, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        # pbar = tqdm(dev_loader)
        with torch.no_grad():
            truth = []
            pred = []
            eval_ct = 0
            for data, label, idx in pbar:
                padding_mask = data != tokenizer.pad_token_id
                output = model(data, padding_mask, labels=label, return_dict=True, )
                graph_embedding = output['graph_embedding'][0]

                # store the graph embedding in the checkpoint
                np.save(os.path.join('checkpoints', args.name, 'graph_embedding_step100.npy'), graph_embedding.detach().cpu().numpy())

                test_loss += output['loss'].item()
                for l in label:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                for l in output['logits']:
                    pred.append(torch.sigmoid(l).tolist())
                eval_ct += 1
                # if eval_ct >= 100:
                #     break
        pbar.close()

        scores = evaluate(pred, truth, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']

        macro_precision = scores['macro_precision']
        micro_precision = scores['micro_precision']
        macro_recall = scores['macro_recall']
        micro_recall = scores['micro_recall']

        print('macro', macro_f1, 'micro', micro_f1)
        print('macro_precision', macro_precision, 'micro_precision', micro_precision)
        print('macro_recall', macro_recall, 'micro_recall', micro_recall)
        print('right_total', scores['right_total'], 'predict_total', scores['predict_total'], 'gold_total', scores['gold_total'])
        import sys
        sys.exit(1)

    model.to(device)
    save = Saver(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))
    log_file = open(os.path.join('checkpoints', args.name, 'log.txt'), 'w')

    for epoch in range(args.max_epoch):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break
        model.train()
        train_ct = 0
        loss = 0
        attention_weights = []

        # Train
        pbar = tqdm(train)
        for data, label, idx in pbar:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, labels=label, return_dict=True, )
            # pred = output['pred'].detach().cpu()
            # compute the f1 score
            graph_embedding = output['graph_embedding'][0]
            if train_ct == 0:
                # store the graph embedding in the checkpoint
                np.save(os.path.join('checkpoints', args.name, 'graph_embedding.npy'), graph_embedding.detach().cpu().numpy())

            truth = []
            pred = []
            for l in label:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                truth.append(t)
            for l in output['logits']:
                pred.append(torch.sigmoid(l).tolist())

            # scores = evaluate(pred, truth, label_dict)
            # macro_f1 = scores['macro_f1']
            # micro_f1 = scores['micro_f1']

            # macro_precision = scores['macro_precision']
            # micro_precision = scores['micro_precision']
            # macro_recall = scores['macro_recall']
            # micro_recall = scores['micro_recall']

            # print('macro', macro_f1, 'micro', micro_f1)
            # print('macro_precision', macro_precision, 'micro_precision', micro_precision)
            # print('macro_recall', macro_recall, 'micro_recall', micro_recall)
            # print('right_total', scores['right_total'], 'predict_total', scores['predict_total'], 'gold_total', scores['gold_total'])

            loss /= args.update
            
            assert len(data) == len(output['attns']), f'{len(data)} != {len(output["attns"])}'
            # detokenize the data
            for i in range(len(data)):
                raw_data = tokenizer.convert_ids_to_tokens(data[i].detach().cpu().tolist())
                
                # get index for 1
                index = torch.nonzero(label[i], as_tuple=True)
                raw_label = [label_dict[v.item()] for v in index[0]]

                # detach the raw_data, raw_label, and attention weight
                # check the device for raw_data and raw_label
                # assert raw_data.device == torch.device('cpu'), f'{raw_data.device} != {torch.device("cpu")}'

                # raw_data = raw_data.detach()
                # raw_label = raw_label.detach()
                attn_weight = output['attns'][i].detach().cpu()
                # check the device for attention weight
                assert attn_weight.device == torch.device('cpu'), f'{attn_weight.device} != {torch.device("cpu")}'
                # attention_weights.append({'raw_data': raw_data, 'label': raw_label,
                #                           'attention_weight': attn_weight})
                # check if the attention weight device
                # assert attn_weight.device == torch.device('cpu'), f'{attn_weight.device} != {torch.device("cpu")}'

            output['loss'].backward()
            loss += output['loss'].item()
            if train_ct % 25 == 0:
                # store the output['label_aware_embedding']
                np.save(os.path.join('checkpoints', args.name, f'label_aware_embedding_{train_ct}.npy'), output['label_aware_embedding'].detach().cpu().numpy())
            train_ct += 1

            if train_ct % args.update == 0:
                optimizer.step()
                optimizer.zero_grad()
                if args.wandb:
                    wandb.log({'train_loss': loss})
                pbar.set_description('loss:{:.4f}'.format(loss))
                # train_ct = 0
                loss = 0
                # torch.cuda.empty_cache()
            # if train_ct > 10:
            #     break
            if train_ct % args.eval_step == 0:
                model.eval()
                # get the top 100 training samples
                
                pbar = tqdm(dev)
                # eval_data = Subset(dataset, split['train'][:1000])
                # dev_loader = DataLoader(eval_data, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

                # pbar = tqdm(dev_loader)
                eval_loss = 0
                with torch.no_grad():
                    truth = []
                    pred = []
                    eval_ct = 0
                    for data, label, idx in pbar:
                        padding_mask = data != tokenizer.pad_token_id
                        output = model(data, padding_mask, labels=label, return_dict=True, )
                        graph_embedding = output['graph_embedding'][0]

                        # store the graph embedding in the checkpoint
                        np.save(os.path.join('checkpoints', args.name, 'graph_embedding_step100.npy'), graph_embedding.detach().cpu().numpy())

                        eval_loss += output['loss'].item()
                        for l in label:
                            t = []
                            for i in range(l.size(0)):
                                if l[i].item() == 1:
                                    t.append(i)
                            truth.append(t)
                        for l in output['logits']:
                            pred.append(torch.sigmoid(l).tolist())
                        eval_ct += 1
                        # if eval_ct >= 100:
                        #     break
                pbar.close()

                scores = evaluate(pred, truth, label_dict)
                macro_f1 = scores['macro_f1']
                micro_f1 = scores['micro_f1']

                macro_precision = scores['macro_precision']
                micro_precision = scores['micro_precision']
                macro_recall = scores['macro_recall']
                micro_recall = scores['micro_recall']

                print('macro', macro_f1, 'micro', micro_f1)
                print('macro_precision', macro_precision, 'micro_precision', micro_precision)
                print('macro_recall', macro_recall, 'micro_recall', micro_recall)
                print('right_total', scores['right_total'], 'predict_total', scores['predict_total'], 'gold_total', scores['gold_total'])
                if args.wandb:
                    wandb.log({'eval_loss': eval_loss / eval_ct})
                    wandb.log(scores)

                pbar.set_description('macro:{:.4f}, micro:{:.4f}'.format(macro_f1, micro_f1))
                model.train()
        pbar.close()

        # attention_weights = attention_weights[-1]

        # generate(attention_weights['raw_data'], attention_weights['attention_weight'], attention_weights['label'], list(label_dict.values()), os.path.join('checkpoints', args.name, f'attention_{str(epoch)}.tex'))
        # convert the attention weights to dataframe and save it
        # attention_weights = pd.DataFrame(attention_weights)
        # attention_weights.to_csv(os.path.join('checkpoints', args.name, 'attention_weights.tsv'), sep='\t')

        # import sys
        # sys.exit(1)

        model.eval()
        pbar = tqdm(dev)
        
        with torch.no_grad():
            truth = []
            pred = []
            eval_loss = 0
            eval_ct = 0
            j = 0
            for data, label, idx in pbar:
                padding_mask = data != tokenizer.pad_token_id
                output = model(data, padding_mask, labels=label, return_dict=True, )
                eval_loss += output['loss'].item()
                for l in label:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                for l in output['logits']:
                    pred.append(torch.sigmoid(l).tolist())
                eval_ct += 1
        pbar.close()
        scores = evaluate(pred, truth, label_dict)

        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']

        macro_precision = scores['macro_precision']
        micro_precision = scores['micro_precision']
        macro_recall = scores['macro_recall']
        micro_recall = scores['micro_recall']

        print('macro', macro_f1, 'micro', micro_f1)
        print('macro_precision', macro_precision, 'micro_precision', micro_precision)
        print('macro_recall', macro_recall, 'micro_recall', micro_recall)
        print('right_total', scores['right_total'], 'predict_total', scores['predict_total'], 'gold_total', scores['gold_total'])
        print('macro', macro_f1, 'micro', micro_f1, file=log_file)
        if args.wandb:
            wandb.log({'eval_loss': eval_loss / eval_ct})
            wandb.log(scores)

        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0
            save_results(pred, truth, scores, new_label_dict, dev_raw_data, epoch, os.path.join('checkpoints', args.name, 'results_best_macro.json'))

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
            save_results(pred, truth, scores, new_label_dict, dev_raw_data, epoch, os.path.join('checkpoints', args.name, 'results_best_micro.json'))

        # save(macro_f1, best_score, os.path.join('checkpoints', args.name, 'checkpoint_{:d}.pt'.format(epoch)))
        # save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
    log_file.close()
