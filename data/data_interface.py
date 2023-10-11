from transformers import AutoTokenizer
from fairseq.data import data_utils
import torch
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pytorch_lightning as pl
import os
import pickle
import math
import json
from tqdm import tqdm
import random

class BertDataset(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None, label_dict=None):
        self.device = device
        super(BertDataset, self).__init__()
        self.data = data_utils.load_indexed_dataset(
            data_path + '/tok', None, 'mmap'
        )
        self.labels = data_utils.load_indexed_dataset(
            data_path + '/Y', None, 'mmap'
        )
        self.max_token = max_token
        self.pad_idx = pad_idx

    def __getitem__(self, item):
        if isinstance(item, int):
            data = self.data[item][:self.max_token - 2].to(
                self.device)
            labels = self.labels[item].to(self.device)
            return {'data': data, 'label': labels, 'idx': item}
        data, label = [], []
        for i in item:
            data.append(self.data[i][:self.max_token - 2])
            label.append(self.labels[i])

        # data = self.data[item][:self.max_token - 2].to(
        #     self.device)
        # labels = self.labels[item].to(self.device)
        return {'data': data, 'label': label, 'idx': item}

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx
    
    def collate_fn_1(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        batch = batch[0]

        label = torch.stack(batch['label'], dim=0)
        data = torch.full([len(batch['data']), self.max_token], self.pad_idx, device=label.device, dtype=batch['data'][0].dtype)
        idx = batch['idx']
        for i, b in enumerate(batch['data']):
            data[i][:len(b)] = b

        return data, label, idx
    
    def get_label(self, idx):
        return self.labels[idx]
    
def get_leaf(labels, label_path):
    leaf = set()
    for label in labels:
        leaf = leaf - set(label_path[label])
        leaf.add(label)
    return list(leaf)

class BertDataset_rcv(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None, label_dict=None, is_test=False,
                 label_path=None):
        self.device = device
        super(BertDataset_rcv, self).__init__()
        
        self.max_token = max_token
        self.pad_idx = pad_idx
        self.is_test = is_test
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        data = []
        labels = []

        num_lines = sum(1 for line in open(data_path,'r'))

        total = 0
        label_hier = {}
        sample2leaf = {}
        # load the json file of data_path
        with open(data_path, 'r') as f:
            if 'wos' in data_path:
                token_key = 'doc_token'
                label_key = 'doc_label'
            else:
                token_key = 'token'
                label_key = 'label'

            for line in tqdm(f, total=num_lines):
                total += 1
                if (not self.is_test) and ('rcv' in data_path) and (total > 100000):
                    break
                line = json.loads(line)
                data.append(tokenizer.encode(line[token_key], truncation=True, max_length=self.max_token, padding='max_length', add_special_tokens=True))


                one_hot = np.zeros(len(label_dict))
                for label in line[label_key]:
                    one_hot[label_dict[label]] = 1
                labels.append(one_hot)
                leaves = get_leaf(line[label_key], label_path)
                sample2leaf[total - 1] = leaves

                for leaf in leaves:
                    path = label_path[leaf]
                    top_label = True
                    top_dict = None
                    for label in path:
                        if top_label:
                            if label not in label_hier:
                                label_hier[label] = {}
                            top_dict = label_hier[label]
                            top_label = False
                        else:
                            if label not in top_dict:
                                top_dict[label] = {}
                            top_dict = top_dict[label]
                    top_dict[total - 1] = 1

        self.label_path = label_path
        self.label_hier = label_hier
        self.sample2leaf = sample2leaf        # load the json file of data_path
        self.data = torch.from_numpy(np.array(data))
        self.labels = torch.from_numpy(np.array(labels))


    def __getitem__(self, item):
        if isinstance(item, int):
            data = self.data[item].to(
                self.device)
            labels = self.labels[item].to(self.device)
            return {'data': data, 'label': labels, 'idx': item}
        data, label = [], []
        for i in item:
            data.append(self.data[i])
            label.append(self.labels[i])

        # data = self.data[item][:self.max_token - 2].to(
        #     self.device)
        # labels = self.labels[item].to(self.device)
        return {'data': data, 'label': label, 'idx': item}

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx
    
    def collate_fn_1(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        batch = batch[0]

        label = torch.stack(batch['label'], dim=0)
        data = torch.full([len(batch['data']), self.max_token], self.pad_idx, device=label.device, dtype=batch['data'][0].dtype)
        idx = batch['idx']
        for i, b in enumerate(batch['data']):
            data[i][:len(b)] = b

        return data, label, idx
    
    def get_label(self, idx):
        return self.labels[idx]
    
    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        #all sub trees end with an int index
        while type(curr_dict) is not int:
            if top_level:
                random_label = label
                if len(curr_dict.keys()) != 1:
                    while (random_label == label):
                        random_label = random.sample(curr_dict.keys(), 1)[0]
            else:
                random_label = random.sample(curr_dict.keys(), 1)[0]
            
            if random_label not in curr_dict:
                random_label = random.sample(curr_dict.keys(), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return curr_dict
    
    def get_split_by_index(self, idx):
        leaves = self.sample2leaf[idx]
        label_by_level = {}
        for leaf in leaves:
            path = self.label_path[leaf]
            for i, label in enumerate(path):
                if i not in label_by_level:
                    label_by_level[i] = set()
                label_by_level[i].add(label)
        return label_by_level
    
class BertDataset_wos(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None, label_dict=None, is_test=False):
        self.device = device
        super(BertDataset_wos, self).__init__()
        
        self.max_token = max_token
        self.pad_idx = pad_idx
        self.is_test = is_test
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        data = []
        labels = []

        num_lines = sum(1 for line in open(data_path,'r'))

        total = 0
        # load the json file of data_path
        with open(data_path, 'r') as f:
            for line in tqdm(f, total=num_lines):
                total += 1
                line = json.loads(line)
                data.append(tokenizer.encode(line['doc_token'], truncation=True, max_length=self.max_token, padding='max_length', add_special_tokens=True))

                one_hot = np.zeros(len(label_dict))
                for label in line['doc_label']:
                    one_hot[label_dict[label]] = 1
                labels.append(one_hot)
        self.data = torch.from_numpy(np.array(data))
        self.labels = torch.from_numpy(np.array(labels))

    def __getitem__(self, item):
        if isinstance(item, int):
            data = self.data[item].to(
                self.device)
            labels = self.labels[item].to(self.device)
            return {'data': data, 'label': labels, 'idx': item}
        data, label = [], []
        for i in item:
            data.append(self.data[i])
            label.append(self.labels[i])

        # data = self.data[item][:self.max_token - 2].to(
        #     self.device)
        # labels = self.labels[item].to(self.device)
        return {'data': data, 'label': label, 'idx': item}

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx
    
    def collate_fn_1(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        batch = batch[0]

        label = torch.stack(batch['label'], dim=0)
        data = torch.full([len(batch['data']), self.max_token], self.pad_idx, device=label.device, dtype=batch['data'][0].dtype)
        idx = batch['idx']
        for i, b in enumerate(batch['data']):
            data[i][:len(b)] = b

        return data, label, idx
    
    def get_label(self, idx):
        return self.labels[idx]
    
class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size: int,
        drop_last: bool, dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,) -> None:

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch=0
        self.num_replicas = 1
        self.rank = 0
        
        # if num_replicas is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     num_replicas = dist.get_world_size()
        # if rank is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     rank = dist.get_rank()
        # self.num_replicas = num_replicas
        # self.rank = rank
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / \
                self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        print(self.total_size, self.num_replicas, self.batch_size,
              self.num_samples, len(self.dataset), self.rank)

    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=10):
        attempt = 0
        while attempt < num_attempt:
            idx = self.dataset.random_sample(
                label, label_dict)
            if idx is None:
                print(label, label_dict.keys())
                print('None')
            elif idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
        idx = remaining[torch.randint(len(remaining), (1,))]
        visited.add(idx)
        return idx

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch = []
        visited = set()
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        remaining = list(set(indices).difference(visited))
        while len(remaining) > self.batch_size:
            idx = indices[torch.randint(len(indices), (1,))]
            batch.append(idx)
            visited.add(idx)
            label_by_level = self.dataset.get_split_by_index(
                idx)
            for level, labels in label_by_level.items():
                rand_label = random.sample(labels, 1)[0]
                
                label_path = self.dataset.label_path[rand_label]

                label_hier = self.dataset.label_hier

                for i, label in enumerate(label_path):
                    label_hier = label_hier[label]
                idx = self.random_unvisited_sample(
                    rand_label, label_hier, visited, indices, remaining)
                batch.append(idx)
                visited.add(idx)
            remaining = list(set(indices).difference(visited))
            if len(batch) >= self.batch_size:
                drop = set(batch[self.batch_size:])
                batch = batch[:self.batch_size]
                visited = visited.difference(drop)
                yield batch
                batch = []
            remaining = list(set(indices).difference(visited))

        if (len(remaining) > self.batch_size) and not self.drop_last:
            batch.update(list(remaining))
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

        
class DInterface(pl.LightningDataModule):
    def __init__(self, args, tokenizer, label_depths, data_path, device, label_dict, positive_threshold: int = 5,
        hard_negative_threshold: int = 10,
        easy_negative_threshold: int = 30):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.label_depths = label_depths
        self.data_path = data_path
        self.device = device
        self.positive_threshold = positive_threshold
        self.hard_negative_threshold = hard_negative_threshold
        self.easy_negative_threshold = easy_negative_threshold
        self.skip_batch_sampling = args.skip_batch_sampling
        # if 'bgc' in data_path:
        #     self.label_dict = label_dict
        # else:
        self.label_dict = {v: k for k, v in label_dict.items()}

    def setup(self, stage=None):
        # Load data
        # label_dict = torch.load(os.path.join(data_path, 'bert_label_dict.pt'))
        # label_dict = {i: self.tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}

        # with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
        #     new_label_dict = pickle.load(f)
        if 'rcv1' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'rcv1_train_all.json')
            val_data_path = os.path.join(self.data_path, 'rcv1_val_all.json')
            if self.args.test_only:
                test_data_path = os.path.join(self.data_path, 'rcv1_test_all.json')
            else:
                test_data_path = os.path.join(self.data_path, 'rcv1_test.json')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, is_test=self.args.test_only, label_path=self.args.label_path)
            self.dataset = self.train_dataset
        elif 'bgc' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'train_data.jsonl')
            val_data_path = os.path.join(self.data_path, 'dev_data.jsonl')
            test_data_path = os.path.join(self.data_path, 'test_data.jsonl')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dataset = self.train_dataset
        elif 'patent' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'train.jsonl')
            val_data_path = os.path.join(self.data_path, 'valid.jsonl')
            test_data_path = os.path.join(self.data_path, 'test.jsonl')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dataset = self.train_dataset
        elif 'aapd' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'train.jsonl')
            val_data_path = os.path.join(self.data_path, 'val.jsonl')
            test_data_path = os.path.join(self.data_path, 'test.jsonl')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dataset = self.train_dataset

        elif 'wos' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'wos_train.json')
            val_data_path = os.path.join(self.data_path, 'wos_val.json')
            test_data_path = os.path.join(self.data_path, 'wos_test.json')

            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dataset = self.train_dataset
        elif 'nyt' in self.data_path:
            train_data_path = os.path.join(self.data_path, 'nyt_train_all.json')
            val_data_path = os.path.join(self.data_path, 'nyt_val_all.json')
            test_data_path = os.path.join(self.data_path, 'nyt_test_all.json')

            # with open('./data/nyt/new_label_dict.pkl', 'rb') as f:
            #     label_dict = pickle.load(f)
            # label_dict = {v : k for k, v in label_dict.items()}
            label_dict = self.label_dict
            self.train_dataset = BertDataset_rcv(data_path=train_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dev_dataset = BertDataset_rcv(data_path=val_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.test_dataset = BertDataset_rcv(data_path=test_data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id, label_dict=label_dict, label_path=self.args.label_path)
            self.dataset = self.train_dataset

        else:
            dataset = BertDataset(data_path=self.data_path, device=self.device, pad_idx=self.tokenizer.pad_token_id)
            self.dataset = dataset

            split = torch.load(os.path.join(self.data_path, 'split.pt'))
            self.train_dataset = Subset(dataset, split['train'])
            self.dev_dataset = Subset(dataset, split['val'])
            self.test_dataset = Subset(dataset, split['test'])

        if not self.skip_batch_sampling:
            self.train_sampler = HierarchicalBatchSampler(batch_size=self.args.batch, dataset=self.train_dataset, drop_last=False,
                                                      )

    def train_dataloader(self):
        if not self.skip_batch_sampling:
            return DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=1, collate_fn=self.dataset.collate_fn_1)
        else:
            return DataLoader(self.train_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
        
    def val_dataloader(self):
        val_dataloader = DataLoader(self.dev_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
        return [val_dataloader, test_dataloader]
        return DataLoader(self.dev_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch, shuffle=False, collate_fn=self.dataset.collate_fn)
    