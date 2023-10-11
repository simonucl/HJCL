from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from .graph import GraphEncoder
from .contrast import SupConLoss, HMLC, WeightedCosineSimilarityLoss
from .bert import BertModel, BertPreTrainedModel, BertEmbeddings, BertPoolingLayer, BertOutputLayer
from .text_attention import generate
from utils import get_hierarchy_info
import pickle
import random
import os

def multilabel_categorical_crossentropy(y_true, y_pred):
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1))
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

# TODO refine the shape
class MultiAttBlock(nn.Module):
    def __init__(self, embed_dim,
                 num_heads,
                 qdim=None,
                 kdim=None,
                 dropout=0.3,
    ):
        # qdim: num_of_labels
        # kdim: seq_len
        super(MultiAttBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_embed_size = qdim if qdim else embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = kdim if kdim else embed_dim
        self.dropout = dropout
        self.head_scale = self.head_dim ** -0.5

        # modify the size here
        d_hq = embed_dim // num_heads
        d_hv = embed_dim // num_heads

        # Define the query, key, and value linear transformations for each head
        # self.query_heads = nn.ModuleList([nn.Linear(self.q_embed_size, self.q_embed_size, bias=False) for _ in range(num_heads)])
        # self.key_heads = nn.ModuleList([nn.Linear(self.k_embed_size, self.k_embed_size, bias=False) for _ in range(num_heads)])
        # self.value_heads = nn.ModuleList([nn.Linear(self.v_embed_size, self.v_embed_size, bias=False) for _ in range(num_heads)])

        self.query_heads = nn.Linear(self.q_embed_size, self.q_embed_size, bias=True)
        self.key_heads = nn.Linear(self.k_embed_size, self.k_embed_size, bias=True)
        self.value_heads = nn.Linear(self.v_embed_size, self.v_embed_size, bias=True)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.query_block = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
    #     for head in self.query_heads:
    #         head.reset_parameters()
    #     for head in self.key_heads:
    #         head.reset_parameters()
    #     for head in self.value_heads:
    #         head.reset_parameters()
        self.query_heads.reset_parameters()
        self.key_heads.reset_parameters()
        self.value_heads.reset_parameters()

        self.out_proj.reset_parameters()
        self.query_block.reset_parameters()

    def multiAttn(self, Q, K, V, key_padding_mask=None):
        # Q: batch_size * num_labels * embed_dim
        # K: batch_size * seq_len * embed_dim
        # V: batch_size * seq_len * embed_dim
        # key_padding_mask: batch_size * seq_len

        Q_proj = self.query_heads(Q) # batch_size * num_labels * embed_dim
        K_proj = self.key_heads(K) # batch_size * seq_len * embed_dim
        V_proj = self.value_heads(V) # batch_size * seq_len * embed_dim

        bsz, num_labels, embed_dim = Q_proj.shape
        bsz, seq_len, embed_dim = K_proj.shape

        Q_proj = Q_proj.transpose(0, 1).reshape(num_labels, bsz * self.num_heads, self.head_dim).transpose(0, 1) # (bsz * num_heads) * num_labels * d_hq
        K_proj = K_proj.transpose(0, 1).reshape(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1) # (bsz * num_heads) * seq_len * d_hq
        V_proj = V_proj.transpose(0, 1).reshape(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1) # (bsz * num_heads) * seq_len * d_hv

        scores = torch.bmm(Q_proj, K_proj.transpose(-2, -1)) * self.head_scale # (bsz * num_heads) * num_labels * seq_len
        
        # check if the scores between different batches are the same
        # print(np.isclose(scores[0].cpu().detach().numpy(), scores[self.num_heads].cpu().detach().numpy()).all())

        if key_padding_mask is not None:
            # Reshape the key_padding_mask to have shape (bsz * num_heads, 1, seq_len) to enable broadcasting
            scores = scores.view(bsz, self.num_heads, num_labels, seq_len)
            # filp the mask
            key_padding_mask = key_padding_mask.eq(0)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # bsz * 1 * 1 * seq_len
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
            scores = scores.view(bsz * self.num_heads, num_labels, seq_len)

        attn_weights = F.softmax(scores, dim=-1) # (bsz * num_heads) * num_labels * seq_len
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, V_proj) # (bsz * num_heads) * num_labels * d_hv
        attn = attn.transpose(0, 1).reshape(num_labels, bsz, self.embed_dim).transpose(0, 1) # bsz * num_labels * embed_dim

        # check if the attention between different batches are the same
        # print(np.isclose(attn[0].cpu().detach().numpy(), attn[1].cpu().detach().numpy()).all())
        attn = self.out_proj(attn) # bsz * num_labels * embed_dim

        attn_weights = attn_weights.view(bsz, self.num_heads, num_labels, seq_len)

        return attn, attn_weights
    
    def forward(self, Q, K, V, key_padding_mask=None, need_weights=False, attn_mask=None):
        # Q: batch_size * num_labels * embed_dim
        # K: batch_size * seq_len * embed_dim
        # V: batch_size * seq_len * embed_dim

        _Q, attns = self.multiAttn(Q, K, V, key_padding_mask=key_padding_mask)
        _Q += self.query_heads(Q)

        return _Q + self.query_block(_Q), attns
        

class LabelAware(nn.Module): # Compute the label aware embedding
    def __init__(self, label_embedding_size, attn_hidden_size, head):
        super(LabelAware, self).__init__()
        self.label_embedding_size = label_embedding_size
        self.attn_hidden_size = attn_hidden_size
        self.multi_attn_block = MultiAttBlock(self.label_embedding_size, head, self.label_embedding_size, self.label_embedding_size)

    def forward(self, input_data, label_repr, input_data_mask=None, label_repr_mask=None):
        # input_data: batch_size * seq_len * hidden_size
        # label_repr: batch_size * num_labels * label_embedding_size
        # input_data_mask: batch_size * seq_len
        
        if input_data_mask is not None:
            input_data = input_data * input_data_mask.unsqueeze(-1)
        if label_repr_mask is not None:
            label_repr = label_repr * label_repr_mask.unsqueeze(-1)
        
        label_aware, attns = self.multi_attn_block(label_repr, input_data, input_data, input_data_mask)

        return label_aware, attns

class Label2Context(nn.Module):  # label attention in the encoder
    def __init__(self, label_embedding_size, attn_hidden_size):
        super(Label2Context,self).__init__()
        self.proj_label = nn.Linear(label_embedding_size, attn_hidden_size, bias=False)

    def forward(self, input_data, label_repr, padding_mask=None):
        # input_data.shape = [batch_size, seq_length, hidden_size]
        # label_repr.shape = [batch_size, num_of_labels, label_embedding_size]
        # padding_mask.shape = [batch_size, seq_length]
        label_repr = self.proj_label(label_repr)
        # label_repr.shape = [batch_size, num_of_labels, hidden_size]

        embedding_label = label_repr.transpose(1, 2)
        # embedding_label.shape = [batch_size, num_of_labels, hidden_size]
        # embedding_label = embedding_label.transpose(1, 2)

        # input_data.shape = [batch_size, seq_length, hidden_size]
        input_data = F.normalize(input_data, dim=-1, p=2)
        # embedding_label.shape = [batch_size, hidden_size, num_of_labels]
        embedding_label = F.normalize(embedding_label, dim=1, p=2)

        # G.shape = [batch_size, seq_length, num_of_labels]
        G = torch.bmm(input_data, embedding_label)

        if padding_mask is not None:
            padding_mask = padding_mask.eq(0)
            G = G.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        # G.shape = [batch_size, seq_length, num_of_labels]
        softmax_G = torch.softmax(G, dim=1) # [batch_size, seq_length, num_of_labels]
        output = torch.bmm(softmax_G.transpose(1, 2), input_data) # [batch_size, num_of_labels, hidden_size]

        return output, softmax_G
        
class LSAN(nn.Module):  # label attention in the encoder
    def __init__(self, label_embedding_size, attn_hidden_size):
        super(LSAN,self).__init__()
        self.proj_label = nn.Linear(label_embedding_size, attn_hidden_size, bias=False)

    def forward(self, input_data, label_repr, padding_mask=None):
        # input_data.shape = [batch_size, seq_length, hidden_size]
        # label_repr.shape = [batch_size, num_of_labels, label_embedding_size]
        # padding_mask.shape = [batch_size, seq_length]

        attn_weights = torch.bmm(label_repr, input_data.transpose(1, 2)) # [batch_size, num_of_labels, seq_length]

        if padding_mask is not None:
            padding_mask = padding_mask.eq(0)
            attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(1), float('-inf'))

        label_attn = torch.bmm(attn_weights, input_data) # [batch_size, num_of_labels, hidden_size]

        return label_attn, attn_weights
    
class LPAN(nn.Module):
    def __init__(self, hidden_size, intermedia_size, num_heads):
        super(LPAN,self).__init__()
        self.hidden_size = hidden_size
        self.intermedia_size = intermedia_size
        self.num_heads = num_heads

        self.f1 = nn.Linear(hidden_size, intermedia_size, bias=False)
        self.f2 = nn.Linear(intermedia_size, num_heads, bias=False)
        self.tanh = nn.Tanh()
        self.proj = nn.Linear(hidden_size * num_heads, hidden_size, bias=False)

        self.u = nn.Linear(hidden_size, intermedia_size, bias=False)
        self.v = nn.Linear(hidden_size, intermedia_size, bias=False)

    def feature_extract(self, input, input_mask=None):
        """
        input: [seq_len, hidden_size]
        """
        f1 = self.f1(input)
        f1 = self.tanh(f1)
        f2 = self.f2(f1)
        
        if input_mask is not None:
            input_mask = input_mask.eq(0)
            f2 = f2.masked_fill(input_mask.unsqueeze(-1), float('-inf'))

        A = torch.softmax(f2, dim=-2) # [seq_len, num_heads]
        M = torch.bmm(A.transpose(-2, -1), input) # [num_heads, hidden_size]

        if len(M.size()) > 2:
            M = M.view(M.size(0), -1)
        elif len(M.size()) == 2:
            M = M.flatten()
        return self.proj(M) # [hidden_size]
    
    def forward(self, input_data, label_repr, input_padding=None, label_padding=None):
        """
        :param input_data: [batch_size, seq_len, hidden_size]
        :param label_repr: [num_of_labels, seq_len, hidden_size]
        :param input_padding: [batch_size, seq_len]
        :param label_padding: [batch_size, num_of_labels]
        """
        input_feature = self.feature_extract(input_data, input_padding) # [batch_size, hidden_size]
        label_feature = self.feature_extract(label_repr, label_padding) # [num_of_labels, hidden_size]

        intermedia_input = self.u(input_feature) # [batch_size, intermedia_size]
        intermedia_label = self.v(label_feature) # [num_of_labels, intermedia_size]

        # do element-wise multiplication to form [num_of_labels, batch_size, intermedia_size]
        intermedia_input = intermedia_input.unsqueeze(0).repeat(intermedia_label.size(0), 1, 1)
        intermedia_label = intermedia_label.unsqueeze(1).repeat(1, intermedia_input.size(1), 1)
        intermedia = intermedia_input * intermedia_label

        # sum up the last dimension to form [num_of_labels, batch_size]
        intermedia = intermedia.sum(dim=-1)
        
class Decoder(nn.Module):
    def __init__(self, hidden_size, graph, label_cpt, tau):
        super().__init__()
        self.hidden_size = hidden_size
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(label_cpt)
        data_path = '/'.join(label_cpt.split('/')[:-1])
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)

        label_depth = [label_depth[v] for k, v in label_dict.items()]
        label_depth = np.array(label_depth, dtype=np.int32)

        self.hiera = hiera
        self.r_hiera = r_hiera
        self.label_depth = label_depth
        # TODO make sure the label dict is in the same order
        self.label_dict = _label_dict

        # create classifier for each layer based on the number of labels at each layer
        depth_counts = Counter(list(label_depth))
        self.classifiers1 = nn.ModuleList()
        self.classifiers2 = nn.ModuleList()
        self.predictors = nn.ModuleList()
        self.depth_counts = depth_counts
        for depth in range(1, max(label_depth)+1):
            self.classifiers1.append(nn.Linear(hidden_size * depth_counts[depth], hidden_size, bias=False))
            self.classifiers2.append(nn.Linear(hidden_size * (depth_counts[depth] + 2), hidden_size, bias=False))
            self.predictors.append(nn.Linear(hidden_size, depth_counts[depth]))
        # self.rnn = nn.GRU(enc_hidden_size+100, dec_hidden_size)
        # self.proj_out = nn.Linear(dec_hidden_size*2+100, dec_hidden_size)

    def seq(self, input, hidden, encoder_outputs):
        """ 
        :param input: [batch_size, hidden_size]
        :param hidden: [batch_size, hidden_size]
        :param encoder_outputs: [batch_size, label_num_at_level, hidden_size]
        """
        attention_outputs = encoder_outputs
        rnn_input = torch.cat([attention_outputs, input], dim=-1) # 
        output, hidden = self.rnn(rnn_input.unsqueeze(0), hidden.unsqueeze(0))

        assert (output == hidden).all()
        output = output.squeeze(0)
        hidden = hidden.squeeze(0)
        pred = self.proj_out(torch.cat([attention_outputs, output, input], dim=1))
        return pred, hidden
    
    def forward(self, label_aware_embedding, proj_label_repr):
        """
        # :param input: [batch_size, seq_length, hidden_size]
        # :param cls: [batch_size, hidden_size]
        # :param label_repr: [1, num_of_labels, hidden_size]
        :param label_aware_embedding: [batch_size, num_of_labels, hidden_size]
        :param proj_label_repr: [batch_size, num_of_labels, hidden_size]
        """
        # check the length of the label_dict equals to the label_repr and label_aware_embedding
        # assert len(self.label_dict) == label_repr.shape[1] == label_aware_embedding.shape[1], 'label dict length not match'

        # create a torch tensor with shape (batch_size, num_of_labels)
        pred = torch.zeros(label_aware_embedding.shape[0], len(self.label_dict)).to(label_aware_embedding.device)
        
        input = None
        hidden = None

        max_depth = max(self.label_depth)
        for i in range(max_depth):
            depth = i + 1
            depth_indices = [idx for idx, d in enumerate(list(self.label_depth)) if d == depth]
            assert len(depth_indices) == self.depth_counts[depth], 'depth indices not match'
            # get the current depth label_aware_embedding
            cur_depth_label_aware_embedding = label_aware_embedding[:, depth_indices, :] # [batch_size, num_of_labels, hidden_size]
            cur_depth_label_aware_embedding = cur_depth_label_aware_embedding.reshape(cur_depth_label_aware_embedding.shape[0], -1) # [batch_size, num_of_labels * hidden_size]
            cur_depth_proj_label_repr = proj_label_repr[:, depth_indices, :] # [batch_size, num_of_labels, hidden_size]
            cur_depth_proj_label_repr = cur_depth_proj_label_repr.reshape(cur_depth_proj_label_repr.shape[0], -1) # [batch_size, num_of_labels * hidden_size]

            # get the current depth label_repr
            # cur_depth_labels = [self.label_dict[idx] for idx in depth_indices]
            
            # apply classifier1
            x = self.classifiers1[i](cur_depth_label_aware_embedding) # [batch_size, hidden_size]
            # if hidden is None:
            #     hidden = x # [batch_size, hidden_size]
            # # concat hidden, x, cur_depth_proj_label_repr
            # x = torch.cat([hidden, x, cur_depth_proj_label_repr], dim=-1) # [batch_size, hidden_size * (depth_counts[depth] + 2)]
            # # apply classifier2
            # x = self.classifiers2[i](x) # [batch_size, hidden_size]
            # hidden = x # [batch_size, hidden_size]
            # apply predictor
            x = self.predictors[i](x) # [batch_size, depth_counts[depth]]
            
            # set the current depth label prediction
            pred[:, depth_indices] = x # [batch_size, num_of_labels]

        return pred
    
class Decoder_GRU(nn.Module):
    def __init__(self, hidden_size, label_cpt, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(label_cpt)
        with open('./data/nyt/new_label_dict.pkl', 'rb') as f:
            label_dict = pickle.load(f)

        label_depth = [label_depth[v] for k, v in label_dict.items()]
        label_depth = np.array(label_depth, dtype=np.int32)

        self.hiera = hiera
        self.r_hiera = r_hiera
        self.label_depth = label_depth
        # TODO make sure the label dict is in the same order
        self.label_dict = _label_dict

        self.rnn = nn.GRU(enc_hidden_size, dec_hidden_size)

        # create classifier for each layer based on the number of labels at each layer
        depth_counts = Counter(list(label_depth))
        self.label_repr_proj = nn.ModuleList()
        # self.classifiers2 = nn.ModuleList()
        self.predictors = nn.ModuleList()
        self.depth_counts = depth_counts
        for depth in range(1, max(label_depth)+1):
            self.label_repr_proj.append(nn.Linear(hidden_size * depth_counts[depth], hidden_size, bias=False))
            # self.classifiers2.append(nn.Linear(hidden_size * (depth_counts[depth] + 2), hidden_size, bias=False))
            self.predictors.append(nn.Linear(hidden_size, depth_counts[depth]))
        # self.rnn = nn.GRU(enc_hidden_size+100, dec_hidden_size)
        # self.proj_out = nn.Linear(dec_hidden_size*2+100, dec_hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def seq(self, input, hidden, encoder_outputs):
        """ 
        :param input: [batch_size, hidden_size]
        :param hidden: [batch_size, hidden_size]
        :param encoder_outputs: [batch_size, label_num_at_level, hidden_size]
        """
        attention_outputs = encoder_outputs
        rnn_input = torch.cat([attention_outputs, input], dim=-1) # 
        output, hidden = self.rnn(rnn_input.unsqueeze(0), hidden.unsqueeze(0))

        assert (output == hidden).all()
        output = output.squeeze(0)
        hidden = hidden.squeeze(0)
        pred = self.proj_out(torch.cat([attention_outputs, output, input], dim=1))
        return pred, hidden
    
    def forward(self, label_aware_embedding, cls):
        """
        # :param input: [batch_size, seq_length, hidden_size]
        # :param cls: [batch_size, hidden_size]
        # :param label_repr: [1, num_of_labels, hidden_size]
        :param label_aware_embedding: [batch_size, num_of_labels, hidden_size]
        :param proj_label_repr: [batch_size, num_of_labels, hidden_size]
        """
        # check the length of the label_dict equals to the label_repr and label_aware_embedding
        # assert len(self.label_dict) == label_repr.shape[1] == label_aware_embedding.shape[1], 'label dict length not match'

        # create a torch tensor with shape (batch_size, num_of_labels)
        pred = torch.zeros(label_aware_embedding.shape[0], len(self.label_dict)).to(label_aware_embedding.device)
        
        input = None
        hidden = None

        max_depth = max(self.label_depth)
        for i in range(max_depth):
            depth = i + 1
            depth_indices = [idx for idx, d in enumerate(list(self.label_depth)) if d == depth]
            assert len(depth_indices) == self.depth_counts[depth], 'depth indices not match'
            # get the current depth label_aware_embedding
            cur_depth_label_aware_embedding = label_aware_embedding[:, depth_indices, :] # [batch_size, num_of_labels, hidden_size]
            cur_depth_label_aware_embedding = cur_depth_label_aware_embedding.reshape(cur_depth_label_aware_embedding.shape[0], -1) # [batch_size, num_of_labels * hidden_size]

            # get the current depth label_repr
            # cur_depth_labels = [self.label_dict[idx] for idx in depth_indices]
            
            # apply classifier1
            x = self.label_repr_proj[i](cur_depth_label_aware_embedding) # [batch_size, hidden_size]
            if hidden is None:
                hidden = cls # [batch_size, hidden_size]
            else:
                hidden = hidden + cls # [batch_size, hidden_size]

            if input is None:
                input = label_aware_embedding[:, depth_indices, :].mean(dim=1) # [batch_size, hidden_size]

            # concat hidden, x, cur_depth_proj_label_repr
            x = torch.cat([x, input], dim=1) # [batch_size, hidden_size * 2]

            output, hidden = self.rnn(x.unsqueeze(0), hidden.unsqueeze(0)) # [1, batch_size, hidden_size], [1, batch_size, hidden_size]
            output = output.squeeze(0) # [batch_size, hidden_size]
            hidden = hidden.squeeze(0) # [batch_size, hidden_size]

            x = self.predictors[i](output) # [batch_size, num_of_labels]
            
            # set the current depth label prediction
            pred[:, depth_indices] = x # [batch_size, num_of_labels]

            # TODO Think of not using softmax here and apply max pooling afterward when computing input
            t = self.softmax(x).unsqueeze(-1).expand(-1, -1, cur_depth_label_aware_embedding.shape[-1]) # [batch_size, num_of_labels, hidden_size]
            input = torch.sum(t * cur_depth_label_aware_embedding, dim=1) # [batch_size, hidden_size]
            
        return pred
    
class ContrastModel(BertPreTrainedModel):
    def __init__(self, config, batch_size=1, cls_loss=True, contrast_loss=True, contrast_mode='label_aware', graph=False, layer=1, data_path=None,
                 multi_label=False, lamb=1, lamb_1=0.1, threshold=0.01, tau=1, device="cuda", head=4, label_cpt=None, label_depths=None, label_dict=None, label_aware_embedding=None,
                 is_decoder=False, softmax_entropy=False, add_reg=True, add_count=False, count_weight=0, do_simple_label_contrastive=False, do_weighted_label_contrastive=False,
                 new_label_dict=None, add_path_reg=False, path_reg_weight=0, path_reg_weight_adjusted=False, ignore_path_reg=False, hamming_dist_mode=None):
        super(ContrastModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.is_decoder = is_decoder
        self.softmax_entropy = softmax_entropy
        
        # TODO modify the classifier for flat multi label or sequence multi label
        if 'patent' not in data_path:
            self.classifier = nn.Linear(config.hidden_size * self.num_labels, self.num_labels)
            self.classifier1 = nn.Linear(config.hidden_size * self.num_labels, config.hidden_size)
            self.classifier2 = nn.Linear(config.hidden_size, self.num_labels)
        else:
            with open(os.path.join(data_path, 'section_dict.pkl'), 'rb') as f:
                section_dict = pickle.load(f)
            self.section_dict = section_dict
            self.classifier1 = {}
            self.classifier2 = {}
            for k, v in section_dict.items():
                self.classifier1[k] = nn.Linear(config.hidden_size * len(v), config.hidden_size)
                self.classifier2[k] = (nn.Linear(config.hidden_size, len(v)))


        
        # self.count_classifier = nn.Linear(config.hidden_size, self.num_labels, bias=True)
        # self.path_proj = nn.Linear(config.hidden_size, 1)

        self.bert = BertModel(config)
        self.pooler = BertPoolingLayer(config, 'cls')
        self.batch_size = batch_size


        self.fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gelu = nn.GELU()

        self.cls_loss = cls_loss
        self.contrast_loss = contrast_loss
        self.contrast_mode = contrast_mode

        if self.contrast_mode == 'straight_through':
            self.straight_fc = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # self.token_classifier = BertOutputLayer(config)
        self.label_aware = LabelAware(config.hidden_size, config.hidden_size, head=head)
        self.label2context = Label2Context(config.hidden_size, config.hidden_size)
        self.LSAN = LSAN(config.hidden_size, config.hidden_size)

        self.add_reg = add_reg

        self.graph_encoder = GraphEncoder(config, graph, layer=layer, data_path=data_path, threshold=threshold, tau=tau, label_dict=new_label_dict)
        # self.decoder = Decoder(config.hidden_size, graph, label_cpt=label_cpt, tau=tau)
        # self.decoder = Decoder_GRU(config.hidden_size, label_cpt=label_cpt, enc_hidden_size=config.hidden_size * 2, dec_hidden_size=config.hidden_size)
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(label_cpt)
        # data_path = '/'.join(data_path.split('/')[:-1])

        if not 'bgc' in data_path:
            with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
                label_dict = pickle.load(f)
        else:
            label_dict = new_label_dict

        if ('nyt' in data_path):
            label_depth = [label_depth[v] for k, v in label_dict.items()]
        elif ('rcv' in data_path) or ('bgc' in data_path):
            label_depth = [label_depth[k] for k, v in _label_dict.items()]
        else:
            label_depth = [label_depth[k] for k, v in label_dict.items()]
        label_depth = np.array(label_depth, dtype=np.int32)

        self.supcon = SupConLoss(temperature=tau, base_temperature=tau, device=device)
        self.data_path = data_path

        self.hiera = hiera
        self.r_hiera = r_hiera
        self.label_depth = label_depth
        # self.hamming_dist = self.compute_hamming_dist(label_dict, r_hiera)
        self.label_dict = label_dict
        self.r_label_dict = {v: k for k, v in label_dict.items()}
        # self.decoder = Decoder_GRU(config.hidden_size, label_cpt=label_cpt, enc_hidden_size=config.hidden_size * 2, dec_hidden_size=config.hidden_size)
        
        # self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, 1, bias=False) for _ in range(self.num_labels)])
        self.add_count = add_count
        self.count_weight = count_weight

        if self.contrast_mode == 'attentive' or self.contrast_mode == 'simple_contrastive':
            self.contrast_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True)
        self.lamb = lamb
        self.lamb_1 = lamb_1

        self.do_simple_label_contrastive = do_simple_label_contrastive
        self.do_weighted_label_contrastive = do_weighted_label_contrastive
        self.hamming_dist_mode = hamming_dist_mode

        self.init_weights()
        self.multi_label = multi_label
        self.tau = tau # temperature
        self.tanh = nn.Tanh()
        # # freeze the bert model
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # for param in self.pooler.parameters():
        #     param.requires_grad = False

        if ('nyt' in data_path):
            self.label_path = {k: self.get_path(v) for k, v in label_dict.items()}
        elif ('rcv' in data_path):
            self.label_path = {k: self.get_path(k) for k, v in _label_dict.items()}
        elif ('bgc' in data_path):
            self.label_path = {v: self.get_path(k) for k, v in _label_dict.items()}
        else:
            self.label_path = {v: self.get_path(k) for k, v in label_dict.items()}
        # self.label_path = {k: self.get_path(v) for k, v in label_dict.items()}
        depth_label_path = {}
        for label in self.label_path:
            depth = len(self.label_path[label])
            if depth not in depth_label_path:
                depth_label_path[depth] = {}
            depth_label_path[depth][label] = self.label_path[label]
        self.depth_label_path = depth_label_path
        self.add_path_reg = add_path_reg
        self.path_reg_weight = path_reg_weight
        self.path_reg_weight_adjusted = path_reg_weight_adjusted
        self.ignore_path_reg = ignore_path_reg

    def hamming_distance_by_matrix(self, labels):
        return torch.matmul(labels, (1 - labels).T) + torch.matmul(1 - labels, labels.T), None
    
    def hamming_distance_by_matrix_weighted_by_depth(self, labels, label_depth):
        # labels: batch_size, num_labels
        # label_depth: num_labels -> depth
        # make it more efficient and times the label_depth[i]
        depths = torch.tensor(label_depth, dtype=torch.float32, device=self.device)
        depths = torch.exp(1 / (depths))
        return torch.matmul(labels * depths, (1 - labels).T) + torch.matmul((1 - labels) * depths, labels.T), torch.sum(depths)
    
    def hamming_distance_by_matrix_weighted_by_depth_1(self, labels, label_depth):
        # labels: batch_size, num_labels
        # label_depth: num_labels -> depth
        # make it more efficient and times the label_depth[i]
        depths = torch.tensor(label_depth, dtype=torch.float32, device=self.device)
        depths = torch.max(depths) - depths + 1
        # depths = torch.exp(1 / (torch.max(depths) - depths + 1))
        return torch.matmul(labels * depths, (1 - labels).T) + torch.matmul((1 - labels) * depths, labels.T), torch.sum(depths)

    def label_contrastive_loss(self, label_embeddings, gold_labels, batch_idx, hamming_dist, depth_sum):
        # label_embeddings: (Positive samples) * hidden_size
        # gold_labels: Positive, ranged from 0 to num_labels
        # batch_idx: Positive samples, record the batch idx of each label embeddings
        # hamming_dist: batch_size * batch_size, hamming distance between batches
        
        # create a expanded hamming distance matrix with the same size with batch_idx
        expanded_hamming_dist = hamming_dist[batch_idx, :][:, batch_idx]

        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        def exp_cosine_sim(x1, x2, eps=1e-15, temperature=1.0):
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = x2.norm(p=2, dim=1, keepdim=True)
            return torch.exp(
                torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)
            )
        
        def exp_sim(x1, x2, eps=1e-15, temperature=1):
            return torch.exp(
                torch.matmul(x1, x2.t()) / temperature
            )
        
        def cosine_sim(x1, x2, eps=1e-15, temperature=1):
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = x2.norm(p=2, dim=1, keepdim=True)
            return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)
            
        for i in range(self.num_labels):
            # get the positive samples
            pos_idx = (gold_labels == i).nonzero().squeeze(1)
            if pos_idx.numel() == 0:
                continue
            # get the negative samples
            neg_idx = (gold_labels != i).nonzero().squeeze(1)
            if neg_idx.numel() == 0:
                continue
            
            pos_samples = label_embeddings[pos_idx] # shape: (num_pos, hidden_size)
            neg_samples = label_embeddings[neg_idx] # shape: (num_neg, hidden_size)
            size = neg_samples.size(0) + 1

            pos_weight = 1 - expanded_hamming_dist[pos_idx, :][:, pos_idx] / depth_sum # shape: (num_pos, num_pos)
            neg_weight = expanded_hamming_dist[pos_idx, :][:, neg_idx] # shape: (num_pos, num_neg)
            pos_dis = exp_cosine_sim(pos_samples, pos_samples) * pos_weight
            neg_dis = exp_cosine_sim(pos_samples, neg_samples) * neg_weight

            denominator = neg_dis.sum(1) + pos_dis
            loss += torch.mean(torch.log(denominator / (pos_dis * size)))
        loss = loss / self.num_labels
        
        return loss
    
        # return: loss

    def compute_hamming_dist(self, label_dict, r_hiera):
        path_labels = torch.zeros((len(label_dict), len(label_dict)), dtype=torch.float32)
        r_label_dict = {v: k for k, v in label_dict.items()}
        for idx, label in label_dict.items():
            while label != 'Root':
                label_idx = r_label_dict[label]
                path_labels[idx][label_idx] = 1

                label = r_hiera[label]
        return torch.matmul(path_labels, (1 - path_labels).T) + torch.matmul(1 - path_labels, path_labels.T)
        
    def get_label_path(self, labels, max_width=4):
        # labels: batch_size * num_labels
        # return: batch_size * max_width
        
        # max_width equals to the number of 1 in self.label_depth
        max_width = np.sum([1 for k, v in enumerate(list(self.label_depth)) if v == 1])
        batch_size = labels.shape[0]

        # fill the label_leaf by -1
        label_leaf = torch.ones((batch_size, max_width), dtype=torch.int64) * -1
        for i in range(batch_size):
            visited = set()
            j = 0
            label_depth = [(self.label_dict[idx],  self.label_depth[idx], idx) for idx, is_label in enumerate(labels[i]) if is_label == 1]
            sorted_label = sorted(label_depth, key=lambda x: x[1], reverse=True)
            for label, depth, idx in sorted_label:
                # check if sorted_label is all in the visited
                if j == max_width:
                    break # TODO fix the bug about j > max_width
                if (set(sorted_label).issubset(visited)) and (visited.issubset(set(sorted_label))):
                    break
                if label in visited:
                    continue
                else:
                    label_leaf[i][j] = idx
                    visited.add(label)
                    j += 1

                while (self.r_hiera[label] not in visited):
                    visited.add(self.r_hiera[label])
        return label_leaf

    def sample_path(self, leaf):
        # leaf: max_width
        # return: (max_width * 2),(max_width * 2)
        max_width = leaf.shape[0]
        path = torch.zeros((max_width * 2), dtype=torch.int64)
        gold = torch.zeros((max_width * 2), dtype=torch.int64)
        for i in range(max_width):
            if (leaf[i] == -1) or (self.label_depth[leaf[i]] <= 2):
                path[2 * i] = -1
                path[2 * i + 1] = -1
                gold[2 * i] = 0
                gold[2 * i + 1] = 0
            else:
                path[2 * i] = leaf[i]
                gold[2 * i] = 1
                
                same_path_label = [k for k, v in enumerate(list(self.label_depth)) if (v == self.label_depth[leaf[i]]) and (k != leaf[i])]
                path[2 * i + 1] = random.choice(same_path_label)
                gold[2 * i + 1] = 0

        return path, gold

    def get_label_path_embedding(self, leaf_idx, label_aware_embedding):
        # leaf_idx: batch_size * (max_width * 2)
        # label_aware_embedding: batch_size * num_labels * bert_hidden_size
        # return: batch_size * (max_width * 2)
        batch_size, max_width = leaf_idx.shape
        label_path_embedding = torch.zeros((batch_size, max_width), dtype=torch.float32)

        for i in range(batch_size):
            for j in range(max_width):
                if leaf_idx[i][j] == -1:
                    continue
                else:

                    path_embedding = label_aware_embedding[i][leaf_idx[i][j].item()]
                    label = self.label_dict[leaf_idx[i][j].item()]
                    count = 1
                    while label != 'Root':
                        path_embedding += label_aware_embedding[i][self.r_label_dict[label]]
                        label = self.r_hiera[label]
                        count += 1

                    label_path_embedding[i][j] = self.tanh(self.path_proj(torch.div(path_embedding, count))) # 
        return label_path_embedding 
    
    def get_leaf(self, labels):
        leaf = set()
        for label in labels:
            label = label[0]
            leaf = leaf - set(self.label_path[label.item()])
            leaf.add(label)
        return list(leaf)
    
    def get_path(self, label):
        path = []
        # label_name = label_dict[label]
        while label != 'Root':
            path.insert(0, label)
            label = self.r_hiera[label]
        return path
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # input_ids: batch_size * seq_len
        # labels: batch_size * num_labels
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        contrast_mask = None

        # sent_inputs: batch_size * seq_len * bert_hidden_size
        bert_out = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,
        )

        hidden_last, pooled, hidden_all = bert_out.last_hidden_state, bert_out.pooler_output, bert_out.hidden_states
        hidden_cls, encode_out = hidden_last[:, 0, :], hidden_last[:, 1:, :]
        # check with np.isclose
        
        # graph_inputs: 1 * label_num * bert_hidden_size
        graph_inputs = self.graph_encoder(lambda x: self.bert.embeddings(x)[0])

        # repeat the graph_inputs to match the batch size
        graph_inputs = graph_inputs.repeat(encode_out.shape[0], 1, 1) # [batch_size, num_of_labels, hidden_size]

        sent_inputs_mask = attention_mask[:, 1:]
        
        # label_aware_embedding: batch_size * num_labels * hidden_size
        label_aware_embedding, attns = self.label_aware(encode_out, graph_inputs, sent_inputs_mask)
        # label_aware_embedding, attns = self.LSAN(sent_inputs.last_hidden_state, graph_inputs, sent_inputs_mask)

        # label_aware_embedding, attns = self.label2context(sent_inputs.last_hidden_state, graph_inputs, sent_inputs_mask)
        # attns: batch_size * num_labels * seq_len
 
        loss = 0
        contrastive_loss = None
        contrast_logits = None

        # fc label_aware_embedding: batch_size * num_labels * hidden_size
        # proj_label_embedding = self.fc(label_aware_embedding)
        proj_label_embedding = self.dropout(label_aware_embedding)
        # add non-linearity
        # label_aware_embedding = torch.tanh(label_aware_embedding)
        # add dropout
        # label_aware_embedding = self.dropout(label_aware_embedding)

        # Multi-label classifier (flat)
        # logits.shape = (batch_size, num_labels)
        # logits = self.classifier(label_aware_embedding)
        # logits = self.decoder(encode_out, hidden_cls, graph_inputs[0, :, :], label_aware_embedding)

        if self.contrast_loss:
            if self.contrast_mode == 'label_aware':
                features = [proj_label_embedding[i, j] for i in range(proj_label_embedding.shape[0]) for j in range(self.num_labels) if labels[i, j]]
                features = torch.stack(features).to(self.device) # (batch_size * num_labels, hidden_size)
                features = torch.unsqueeze(features, 1) # (batch_size * num_labels, 1, hidden_size)
            elif self.contrast_mode == 'fusion':
                features = [torch.concat([proj_label_embedding[i, j], graph_inputs[0, j, :].squeeze()], dim=-1) for i in range(proj_label_embedding.shape[0]) for j in range(self.num_labels) if labels[i, j]]
                features = torch.stack(features).to(self.device) # (batch_size * num_labels, hidden_size + bert_hidden_size)
                features = torch.unsqueeze(features, 1) # (batch_size * num_labels, 1, hidden_size + bert_hidden_size)
            elif self.contrast_mode == 'attentive':
                # shape of proj_label_embedding: batch_size * num_labels * hidden_size

                # append the label embedding to the end of the sentence embedding
                # shape of proj_label_embedding: batch_size * num_labels * (hidden_size * 2)
                fusion_label_embedding = torch.cat([proj_label_embedding, graph_inputs], dim=-1) # batch_size * num_labels * (hidden_size * 2)
                fusion_attn_weights = self.contrast_proj(fusion_label_embedding) # batch_size * num_labels * hidden_size
                fusion_attn_weights = torch.softmax(fusion_attn_weights, dim=-1) # batch_size * num_labels * hidden_size
                fusion_attn_weights = torch.bmm(fusion_attn_weights, encode_out.transpose(1, 2)) # batch_size * num_labels * seq_len

                label_specifc_embedding = torch.bmm(fusion_attn_weights, encode_out) # batch_size * num_labels * bert_hidden_size

                features = label_specifc_embedding

            elif self.contrast_mode == 'simple_contrastive':
                fusion_label_embedding = torch.cat([proj_label_embedding, graph_inputs], dim=-1) # batch_size * num_labels * (hidden_size * 2)
                fusion_attn_weights = self.contrast_proj(fusion_label_embedding) # batch_size * num_labels * hidden_size
                fusion_attn_weights = torch.softmax(fusion_attn_weights, dim=-1) # batch_size * num_labels * hidden_size
                fusion_attn_weights = torch.bmm(fusion_attn_weights, encode_out.transpose(1, 2)) # batch_size * num_labels * seq_len

                label_specifc_embedding = torch.bmm(fusion_attn_weights, encode_out) # batch_size * num_labels * bert_hidden_size

                # create mask based on labels which eq 1
                mask = labels.to(torch.bool)
                mask = mask.unsqueeze(-1).expand_as(label_specifc_embedding) # batch_size * num_labels * bert_hidden_size
                label_specifc_embedding = torch.masked_select(label_specifc_embedding, mask).view(-1, label_specifc_embedding.shape[-1]) # (batch_size * num_labels) * bert_hidden_size

                features = label_specifc_embedding
            elif self.contrast_mode == 'straight_through':
                features = proj_label_embedding # batch_size * num_labels * hidden_size

        label_aware_embedding = self.dropout(features)

        if self.contrast_mode == 'straight_through':
            label_aware_embedding = features

        if not self.is_decoder:
        # flatten the label_aware_embedding into batch_size * (num_labels * hidden_size)
            # if self.classifier1 is not list:
            if not isinstance(self.classifier1, dict):
                cls_embedding = label_aware_embedding.view(-1, label_aware_embedding.shape[1] * label_aware_embedding.shape[2])
                # label_aware_embedding = self.dropout(label_aware_embedding)
                ## TODO try to add dropout here
                intermediate_embedding = self.classifier1(cls_embedding)
                intermediate_embedding = torch.relu(intermediate_embedding)
                logits = self.classifier2(intermediate_embedding)
            else:
                logits = np.zeros((label_aware_embedding.shape[0], self.num_labels))
                for k, v in self.section_dict.items():
                    # index the v in label_aware_embedding
                    label_aware_embedding_section = label_aware_embedding[:, v, :]
                    label_aware_embedding_section = label_aware_embedding_section.view(-1, label_aware_embedding_section.shape[1] * label_aware_embedding_section.shape[2])
                    # label_aware_embedding_section = self.dropout(label_aware_embedding_section)
                    intermediate_embedding = self.classifier1[k](label_aware_embedding_section)
                    intermediate_embedding = torch.relu(intermediate_embedding)
                    logits_section = self.classifier2[k](intermediate_embedding)
                    logits[:, v] = logits_section
                    
        # Multi-label classifier (One for each label)
        # label_aware_embedding = self.dropout(label_aware_embedding)
        # logits = [self.classifiers[i](label_aware_embedding[:, i, :]) for i in range(self.num_labels)] # num_labels * (batch_size, 1)
        # logits = torch.cat(logits, dim=1) # batch_size * num_labels
        # logits = self.classifier(label_aware_embedding)
        else:
        # Multi-label decoder (One for each layer)
            cls_embedding = label_aware_embedding
            logits = self.decoder(cls_embedding, cls_embedding) # batch_size * num_labels

        # only when training
        if labels is not None:
            if not self.multi_label:
                loss_fct = CrossEntropyLoss()
                target = labels.view(-1)
            elif self.softmax_entropy:
                target = labels.to(torch.float32)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                target = labels.to(torch.float32) # batch_size * num_labels

            if self.softmax_entropy:
                loss += multilabel_categorical_crossentropy(target, logits.view(-1, self.num_labels))
            else:
                loss += loss_fct(logits.view(-1, self.num_labels), target)

        label_loss = 0
        weighted_label_contrastive_loss = 0
        # copy label_aware_embedding
        if self.do_simple_label_contrastive:
            mask = labels.to(torch.bool) # batch_size * num_labels
            # flat the last two dimensions of label_aware_embedding_2
            label_aware_embedding_2 = label_aware_embedding.view(-1, label_aware_embedding.shape[-1]) # (batch_size * num_labels) * hidden_size
            # get the label_aware_embedding for the labels which eq 1
            mask = mask.unsqueeze(-1).view(-1, 1) # (batch_size * num_labels) * 1
            label_aware_embedding_2 = torch.masked_select(label_aware_embedding_2, mask).view(-1, label_aware_embedding_2.shape[-1]) # (batch_size * num_labels) * hidden_size

            # unsqueeze the label_aware_embedding_2
            label_aware_embedding_2 = label_aware_embedding_2.unsqueeze(1) # (batch_size * num_labels) * 1 * hidden_size

            gold_labels = labels.view(-1) # (batch_size * num_labels)
            # get the gold indices
            gold_labels = torch.nonzero(gold_labels).squeeze() # (batch_size * num_labels)
            # take modulo to get the indices of the gold labels
            gold_labels = gold_labels % self.num_labels # (batch_size * num_labels)

            # compute the simcontrast loss
            label_loss = self.supcon(label_aware_embedding_2, gold_labels).float().to(self.device)
            loss += label_loss * self.lamb_1

            # print("label_loss: ", label_loss)

        if True:
            # compute the hamming distance between the labels
            labels = labels.to(torch.float32) # batch_size * num_labels
            
            if self.hamming_dist_mode is None:
                hamming_dist, _ = self.hamming_distance_by_matrix(labels) # distance between labels, batch_size * batch_size
                depth_sum = torch.tensor(self.num_labels)
            elif self.hamming_dist_mode == "discounted_weight":
                hamming_dist, depth_sum = self.hamming_distance_by_matrix_weighted_by_depth(labels, self.label_depth) # distance between labels, batch_size * batch_size
            elif self.hamming_dist_mode == "depth_weight":
                hamming_dist, depth_sum = self.hamming_distance_by_matrix_weighted_by_depth_1(labels, self.label_depth) # distance between labels, batch_size * batch_size
            
            hamming_dist = hamming_dist.to(self.device)
            depth_sum = depth_sum.to(self.device)

            # create indices tensor that repeat batch_size * 1, batch_size * 2, ..., batch_size * num_labels
            batch_idx = torch.arange(label_aware_embedding.shape[0]).to(self.device) # shape = (batch_size)
            batch_idx = batch_idx.unsqueeze(1).expand(-1, self.num_labels).flatten() # shape = (batch_size * num_labels)

            # flatten the label_aware_embedding into (batch_size * num_labels) * hidden_size
            # label_aware_embedding_2 = self.fc2(self.gelu(self.fc1(label_aware_embedding))) # (batch_size * num_labels) * hidden_size
            label_aware_embedding_2 = label_aware_embedding.view(-1, label_aware_embedding.shape[-1])

            # get the label_aware_embedding for the labels which eq 1
            mask = labels.to(torch.bool) # batch_size, num_labels
            mask = mask.flatten() # (batch_size * num_labels)

            # get the indices of the mask which eq 1
            gold_label = torch.nonzero(mask).squeeze() # (batch_size * num_labels)
            # take moduluo of the self.num_labels to get the indices of the labels
            gold_label = gold_label % self.num_labels # (batch_size * num_labels)

            masked_batch_idx = torch.masked_select(batch_idx, mask) # (batch_size * num_labels)
            masked_label_aware_embedding = torch.masked_select(label_aware_embedding_2, mask.unsqueeze(-1).expand_as(label_aware_embedding_2)).view(-1, label_aware_embedding_2.shape[-1]) # (batch_size * num_labels) * hidden_size

            weighted_label_contrastive_loss = self.label_contrastive_loss(masked_label_aware_embedding, gold_label, masked_batch_idx, hamming_dist, depth_sum).to(self.device)

            # print("Weighted label contrastive loss: ", weighted_label_contrastive_loss)

        if self.do_weighted_label_contrastive:
            loss += weighted_label_contrastive_loss * self.lamb_1


        count_loss = 0
        # Count-based loss
        if (self.training) and (self.add_count):
            if not self.is_decoder:
                count_embedding = self.count_classifier(intermediate_embedding) # batch_size * num_labels
                count_embedding = torch.softmax(count_embedding, dim=-1) # batch_size * num_labels
                count_embedding = torch.log(count_embedding) # batch_size * num_labels

                # create gold_count vector with shape (batch_size, num_labels)
                gold_count = torch.zeros_like(count_embedding)

                # fill each row by the number of 1 in labels vector
                for i in range(labels.shape[0]):
                    gold_count[i, torch.sum(labels[i]).item()] = 1
                # perform hammard product between count_embedding and labels
                count_embedding = count_embedding * gold_count.to(torch.float32) # batch_size * num_labels
                
                count_embedding = -1 * count_embedding
                count_loss = torch.sum(count_embedding, dim=-1) # batch_size
                count_loss = torch.mean(count_loss) # 1
                loss += count_loss * self.count_weight
            else:
                # throw not implemented error
                raise NotImplementedError

        path_reg_loss = 0
        if (self.add_path_reg):
            # need label_dict, r_hiera, depths
            if not self.is_decoder:
                # extract path from target
                target_with_depth = [[(p, self.label_depth[p]) for p in torch.where(t > 0)[0]] for t in target]
                # sort the target_with_depth by depth in 2nd dimension
                target_with_depth = [sorted(t, key=lambda x: x[1]) for t in target_with_depth]

                leaf = [self.get_leaf(t) for t in target_with_depth]

                path_reg_bce_crit = nn.BCEWithLogitsLoss()
                path_reg_bce_crit = path_reg_bce_crit.to(self.device)

                path_reg_loss = 0
                for i, l in enumerate(leaf):

                    reg_loss = 0
                    for label in l:
                        path_reg_target = []
                        path_reg_embed = []

                        # print('Label', label, label.item())

                        depth = self.label_depth[label.item()]
                        # print('Depth', depth)
                        pos_path = self.label_path[label.item()]
                        # pos_path = [pos_path[i] for i in range(len(pos_path))]
                        if ('nyt' in self.data_path) or ('bgc' in self.data_path):
                            pos_path = [self.r_label_dict[i] for i in pos_path] # (num_pos_label)
                        else:
                            pos_path = [self.label_dict[i] for i in pos_path] # (num_pos_label)

                        neg_path = self.depth_label_path[depth] # (num_pos)
                        # print('Neg path', neg_path)
                        neg_path = [neg_path[i] for i in neg_path if i != label] # (num_neg)
                        if ('nyt' in self.data_path) or ('bgc' in self.data_path):
                            neg_path = [[self.r_label_dict[_i] for _i in i] for i in neg_path] # (num_neg, num_neg_labnl)
                        else:
                            neg_path = [[self.label_dict[_i] for _i in i] for i in neg_path]

                        neg_path = torch.from_numpy(np.array(neg_path))

                        # print('Path', pos_path, neg_path)
                        # print(embeddings[pos_path])
                        pos_embedding = logits[i, pos_path].float().mean(axis=0)

                        def get_neg_embedding_weight(pos_path, neg_path):
                            overlap = [len([i for i in neg if i in pos_path]) for neg in neg_path]
                            # print('Overlap', overlap)
                            weights = [[1] * overlap[i] + [(depth)/ (depth - overlap[i])] * (depth - overlap[i]) for i in range(len(neg_path))]
                            return torch.from_numpy(np.array(weights)).to(self.device)
                        if self.path_reg_weight_adjusted:
                            neg_weight = get_neg_embedding_weight(pos_path, neg_path)
                        else:
                            neg_weight = torch.ones_like(neg_path).to(self.device)
                        # print('Neg weight', neg_weight)
                        neg_embedding = torch.mul(logits[i, neg_path], neg_weight).float().mean(axis=1)

                        path_reg_embed.extend(torch.cat([pos_embedding.unsqueeze(0), neg_embedding], dim=0)) # (num_neg + 1, hidden_size)
                        path_reg_target.extend([1] + [0] * neg_embedding.shape[0]) # (num_neg + 1)

                        # take softmax of path_reg_embed
                        reg_loss += path_reg_bce_crit(torch.stack(path_reg_embed).float().to(self.device), torch.tensor(path_reg_target).float().to(self.device))

                    # TODO scale by the depth / the number of negative path
                    # path_reg_loss += multilabel_categorical_crossentropy(torch.tensor(path_reg_target).to(self.device)
                    #                                                      , torch.stack(path_reg_embed).to(self.device))
                    path_reg_loss += reg_loss / len(l)
                    # path_reg_loss = reg_loss

                path_reg_loss = path_reg_loss.to(self.device)
                # path_reg_loss = path_reg_loss / len(leaf)
                
                if not self.ignore_path_reg:
                    loss += path_reg_loss * self.path_reg_weight
            else:
                raise NotImplementedError
            
                # get all the target and non-target path with the same depth

                # calculate the sum of logits of target path and non-target path

                # use multilabel_categorical_crossentropy to calculate the loss


        # Path-embedding loss
        # if (self.training) and (self.add_reg):
        #     # labels.shape = (batch_size, num_labels)
        #     # get the label path for each batch
        #     label_path = self.get_label_path(labels) # batch_size * max_width
        #     max_width = label_path.shape[1]

        #     # create two matrix with shape (batch_size, max_width * 2)
        #     reg_idx = torch.zeros((label_path.shape[0], max_width * 2), dtype=torch.long)
        #     true_label = torch.zeros((label_path.shape[0], max_width * 2), dtype=torch.long)

        #     # fill the matrix
        #     for i in range(label_path.shape[0]):
        #         reg_idx[i], true_label[i] = self.sample_path(label_path[i])

        #     # get the label embedding for each label path
        #     reg_idx_embedding = self.get_label_path_embedding(reg_idx, label_aware_embedding_2) # batch_size * (max_width * 2)
                 
        #     new_loss = multilabel_categorical_crossentropy(true_label, reg_idx_embedding)

        #     print('new_loss: ', new_loss)
        #     # compute the multilabel_categorical_crossentropy loss
        #     loss += new_loss
            # fill by zero
        
                                        
                    
        # check if is training
        if self.training:

            return {
                'loss': loss,
                'logits': logits,
                # 'pred': torch.sigmoid(logits),
                # 'hidden_states': outputs.hidden_states,
                # 'attentions': outputs.attentions,
                # 'contrast_logits': contrast_logits,
                # 'attns': attns,
                # 'graph_embedding': graph_inputs,
                'features': features if self.contrast_mode != 'straight_through' else self.straight_fc(features),
                'count_loss': count_loss,
                'label_loss': label_loss if self.do_simple_label_contrastive else weighted_label_contrastive_loss,
                'path_reg_loss': path_reg_loss,
                'weighted_label_contrastive_loss': weighted_label_contrastive_loss,
            }
        else:
            return {
                'loss': loss,
                'logits': logits,
                # 'pred': torch.sigmoid(logits),
                # 'hidden_states': outputs.hidden_states,
                # 'attentions': outputs.attentions,
                # 'contrast_logits': contrast_logits,
                'attns': attns,
                # 'graph_embedding': graph_inputs,
                'features': features,
                'label_loss': label_loss,
                'path_reg_loss': path_reg_loss,
                'weighted_label_contrastive_loss': weighted_label_contrastive_loss,
            }
