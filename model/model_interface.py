from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import pytorch lightening
import pytorch_lightning as pl
from .model import ContrastModel
from .contrast import SupConLoss, HMLC, WeightedCosineSimilarityLoss
from .optim import Adam, ScheduledOptim
from .eval import evaluate, evaluate_by_level
import os
import pickle 

# TODO check how the device is passed
class MInterface(pl.LightningModule):
    def __init__(self, args, num_labels, label_depths, device, data_path, label_dict, new_label_dict, r_hiera):
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.label_dict = label_dict
        self.contrast_loss = args.contrast
        self.lamb = args.lamb
        self.ignore_contrastive = args.ignore_contrastive
        self.new_label_dict = new_label_dict
        self.r_hiera = r_hiera
        self.model = ContrastModel.from_pretrained(self.model_name, args.batch, num_labels=num_labels,
                                          contrast_loss=args.contrast, contrast_mode= args.contrast_mode,
                                          graph=args.graph, label_depths=label_depths, device=device,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre, tau=args.tau, head=args.head,
                                          label_cpt=args.label_cpt, softmax_entropy=args.softmax_entropy,
                                          do_weighted_label_contrastive=args.do_weighted_label_contrastive, lamb_1=args.lamb_1,
                                          new_label_dict=self.new_label_dict,
                                          hamming_dist_mode=args.hamming_dist_mode)

        
        if args.contrast_mode == 'simple_contrastive':
            self.contrast = SupConLoss(temperature=args.tau, device=device)
        else:
        # self.contrast = SupConLoss(temperature=args.tau, device=device)
            self.contrast = HMLC(temperature=args.tau, layer_penalty=torch.exp, label_depths=torch.Tensor(label_depths).to(device), device=device)
        # self.contrast = WeightedCosineSimilarityLoss(class_num=num_labels)

        self.threshold = 0 if self.args.softmax_entropy else 0.5

        self.loss = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.validation_step_preds = []
        self.validation_step_truth = []
        self.test_step_preds = []
        self.test_step_truth = []
        self.test_outputs = []
 
    def forward(self, input_ids, label, idx):
        padding_mask = input_ids != self.tokenizer.pad_token_id
        outputs = self.model(input_ids, padding_mask, labels=label, return_dict=True)

        return outputs

    def training_step(self, batch, batch_idx):
        input_ids, label, idx = batch
        # label.shape = (batch_size, num_labels)
        self.model.train()

        output = self(input_ids, label, idx)
        loss = output['loss']
        count_loss = output['count_loss']
        label_loss = output['label_loss']
        weighted_label_con_loss = output['weighted_label_contrastive_loss']
        path_reg_loss = output['path_reg_loss']

        # classification_loss = loss.item()
        features = output['features']

        
        if self.args.contrast_mode == 'simple_contrastive':
            new_labels = []
            # get the index for each row in the label matrix
            for i in range(label.shape[0]):
                new_labels.append(torch.where(label[i] == 1)[0])
            label = torch.cat(new_labels)
            
        self.log('classification_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.contrast_loss:
            if self.args.contrast_mode == 'simple_contrastive':
                features = features.unsqueeze(1)
            all_features = self.all_gather(features, sync_grads=True)
            all_label = self.all_gather(label, sync_grads=True)

            # stack the first dimension
            if len(all_features.shape) > 3:
                all_features = torch.cat(all_features.unbind(dim=0), dim=0)
                all_label = torch.cat(all_label.unbind(dim=0), dim=0)

            if self.args.contrast_mode == 'simple_contrastive':
                con_loss = self.contrast(all_features, all_label)
            else:
                con_loss, loss_by_depths = self.contrast(all_features, all_label)
            # con_loss = self.contrast(all_features, all_label)
        
            if (self.current_epoch % 2 == 0) and (batch_idx % 300 == 0):
                with open(os.path.join(self.args.save_path, f'con_loss_{str(self.current_epoch)}_{str(batch_idx)}.pkl'), 'wb') as f:
                    pickle.dump(con_loss, f)
                # with open(os.path.join(self.args.save_path, f'loss_by_depths_{str(self.current_epoch)}_{str(batch_idx)}.pkl'), 'wb') as f:
                #     pickle.dump(loss_by_depths, f)

                # with open(os.path.join(self.args.save_path, f'mask_by_depths_{str(self.current_epoch)}_{str(batch_idx)}.pkl'), 'wb') as f:
                #     pickle.dump(mask_by_depths, f)

                # with open(os.path.join(self.args.save_path, f'all_features_{str(self.current_epoch)}_{str(batch_idx)}.pkl'), 'wb') as f:
                #     pickle.dump(all_features, f)

                # with open(os.path.join(self.args.save_path, f'all_label_{str(self.current_epoch)}_{str(batch_idx)}.pkl'), 'wb') as f:
                #     pickle.dump(all_label, f)

            if (self.args.contrast_mode == 'attentive') or (self.args.contrast_mode == 'straight_through'):
                for i, depth_loss in enumerate(loss_by_depths):
                    # print(1)
                    # print(f'contrast_loss_depth_{i}: {depth_loss.item()}')
                    self.log(f'contrast_loss_depth_{i}', depth_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                    
                self.log('contrast_loss', con_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            else:
            # print(f'contrast_loss: {con_loss.item()}')
                self.log('contrast_loss', con_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            if not self.ignore_contrastive:
                loss += con_loss * self.lamb
            # loss += self.contrast(all_features, all_label)[0] * self.lamb

        self.log('label_loss', label_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('count_loss', count_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('weighted_label_contrastive_loss', weighted_label_con_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('path_reg_loss', path_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        input_ids, label, idx = batch
        self.model.eval()
        output = self(input_ids, label, idx)
        loss = output['loss']

        logits = output['logits'].detach().cpu()
        label = label.detach().cpu()
        for l in label:
            t = []
            for i in range(l.size(0)):
                if l[i].item() == 1:
                    t.append(i)
            if dataloader_idx == 0:
                self.validation_step_truth.append(t)
            else:
                self.test_step_truth.append(t)

        for p in logits:
            if dataloader_idx == 0:
                if self.args.softmax_entropy:
                    self.validation_step_preds.append(p.tolist())
                else:
                    self.validation_step_preds.append(torch.sigmoid(p).tolist())
            else:
                if self.args.softmax_entropy:
                    self.test_step_preds.append(p.tolist())
                else:
                    self.test_step_preds.append(torch.sigmoid(p).tolist())

        if dataloader_idx == 0:        
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        scores = evaluate(self.validation_step_preds, self.validation_step_truth, self.label_dict, self.new_label_dict, self.r_hiera, threshold=self.threshold)
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
        
        # add prefix 'val/' to all keys
        scores = {f'val/{k}': v for k, v in scores.items()}

        self.log_dict(scores)

        self.validation_step_preds.clear()
        self.validation_step_truth.clear()


        scores = evaluate(self.test_step_preds, self.test_step_truth, self.label_dict, self.new_label_dict, self.r_hiera, threshold=self.threshold)
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
        
        # add prefix 'test/' to all keys
        scores = {f'test/{k}': v for k, v in scores.items()}

        self.log_dict(scores)

        self.test_step_preds.clear()
        self.test_step_truth.clear()

    def on_test_epoch_end(self) -> None:
        test_step_preds = []
        test_step_truth = []
        attns = []
        losses = []
        indices = []
        input_ids = []
        labels = []
        preds = []

        outputs = self.test_outputs
        

        # sample 5 random index from outputs size
        rand_idx = np.random.choice(len(outputs), 5)

        # if 'attns' in outputs[0]:
        #     print(outputs[0]['attns'])
        for idx, out in enumerate(outputs):
            logits = out['logits']
            label = out['label']

            # if logits and label is in gpu, move to cpu
            if logits.is_cuda:
                logits = logits.detach().cpu()
            if label.is_cuda:
                label = label.detach().cpu()

            # logits = out['logits'].detach().cpu()
            # label = out['label'].detach().cpu()
            for l in label:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                test_step_truth.append(t)
            for p in logits:
                if self.args.softmax_entropy:
                    test_step_preds.append(p.tolist())
                else:
                    test_step_preds.append(torch.sigmoid(p).tolist())

            # if idx in rand_idx:
            #     attns.append(out['attns'].detach().cpu().numpy())
            #     losses.append(out['loss'].detach().cpu().numpy())
            #     indices.append(out['idx'])
            #     input_ids.append(out['input_ids'].detach().cpu().numpy())
            #     labels.append(out['label'].detach().cpu().numpy())
                
        scores = evaluate(test_step_preds, test_step_truth, self.label_dict, self.new_label_dict, self.r_hiera, threshold=self.threshold)
        # depth_scores = evaluate_by_level(test_step_preds, test_step_truth, self.label_dict, self.new_label_dict, self.r_hiera, threshold=self.threshold, depths=self.args.depths)

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

        # store the attns and loss into file
        # attns = np.concatenate(attns, axis=0)
        # # losses = np.concatenate(losses, axis=0)
        # indices = np.concatenate(indices, axis=0)
        # input_ids = np.concatenate(input_ids, axis=0)

        # store the all_attns, all_loss and all_indices into file
        # if len(self.args.gpus.split(',')) == 1:
        #     with open(os.path.join(self.args.save_path, 'attns.pkl'), 'wb') as f:
        #         pickle.dump(attns, f)
        #     with open(os.path.join(self.args.save_path, 'indices.pkl'), 'wb') as f:
        #         pickle.dump(indices, f)
        #     with open(os.path.join(self.args.save_path, 'input_ids.pkl'), 'wb') as f:
        #         pickle.dump(input_ids, f)
        #     with open(os.path.join(self.args.save_path, 'labels.pkl'), 'wb') as f:
        #         pickle.dump(labels, f)
        # else:
        #     with open(os.path.join(self.args.save_path, 'attns.pkl'), 'ab') as f:
        #         pickle.dump(attns, f)
        #     with open(os.path.join(self.args.save_path, 'indices.pkl'), 'ab') as f:
        #         pickle.dump(indices, f)
        #     with open(os.path.join(self.args.save_path, 'input_ids.pkl'), 'ab') as f:
        #         pickle.dump(input_ids, f)
        #     with open(os.path.join(self.args.save_path, 'labels.pkl'), 'ab') as f:
        #         pickle.dump(labels, f)

        # # scores['label_consistency'] = compute_label_consistency(test_step_preds, test_step_truth, self.label_dict)
        with open(os.path.join(self.args.save_path, 'preds.pkl'), 'wb') as f:
            pickle.dump(test_step_preds, f)
        with open(os.path.join(self.args.save_path, 'truth.pkl'), 'wb') as f:
            pickle.dump(test_step_truth, f)

        with open(os.path.join(self.args.save_path, 'label_dict.pkl'), 'wb') as f:
            pickle.dump(self.label_dict, f)
        # with open(os.path.join(self.args.save_path, 'depth_scores.pkl'), 'wb') as f:
        #     pickle.dump(depth_scores, f)
    
        self.log_dict(scores)
        self.test_outputs.clear()

    def test_step(self, batch, batch_idx):
        input_ids, label, idx = batch
        self.model.eval()

        output = self(input_ids, label, idx)
        loss = output['loss']
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        output['label'] = label
        # output['idx'] = idx
        # output['input_ids'] = input_ids

        output_cpu = {}
        # make the whole dict to be on cpu
        for k, v in output.items():
            if k not in ['logits', 'label']:
                continue
            if isinstance(v, torch.Tensor):
                output_cpu[k] = v.detach().cpu()
            else:
                output_cpu[k] = v
            
        self.test_outputs.append(output_cpu)
        return output
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.args.lr)

        if self.args.warmup > 0:
            def lr_warmup(step):
                if step < self.args.warmup:
                    return step / max(1, self.args.warmup)
                return 1
            
            def warmup_then_decay_schedule(epoch):
                if epoch < self.args.warmup:
                    return (epoch + 1) / self.args.warmup
                elif self.args.decay_epochs == 0:
                    return 1.0
                else:
                    decay_factor = (epoch + 1 - self.args.warmup) / self.args.decay_epochs
                    return (1.0 - decay_factor)
            
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_warmup)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_decay_schedule)
            # return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]
            return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}]
        else:
            return [optimizer]
        
    

    