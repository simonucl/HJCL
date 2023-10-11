from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce', label_depths=None, device='cuda'):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device
        self.label_depths = label_depths # shape (num_labels, 1)
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature, device=device)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        """
        :param features: shape (batch_size, num_labels, feature_dim)
        :param labels: shape (batch_size, num_labels)
        """
        # device = (torch.device(self.device)
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = labels.device
        self.label_depths = self.label_depths.to(device)
        
        # mask = torch.ones(labels.shape).to(device) # shape (batch_size, 4 <- hierarchy)
        mask = labels.clone().detach().to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))
        max_depths = int(torch.max(self.label_depths).item()) + 1
        # capture the loss for each layer by create a list of losses
        loss_by_depths = []
        mask_by_depths = []
        for l in range(1, max_depths):
            # get depth_mask with those smaller than max_depths - l
            depth_mask = self.label_depths <= (max_depths - l) # shape (num_labels, 1)
            # check the device for depth_mask and mask
            assert depth_mask.device == mask.device, f'depth_mask.device: {depth_mask.device}, mask.device: {mask.device}'
            # filter the mask with depth_mask
            mask = mask * depth_mask # shape (batch_size, num_labels)
            # mask[:, labels.shape[1]-l:] = 0
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)

            layer_loss = self.sup_con_loss(features, mask=mask_labels)
            mask_by_depths.insert(0, mask_labels.detach().cpu().numpy())

            # l = l+1
            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                  l).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                tmp = torch.tensor(1 / (l)).type(torch.float).to(layer_loss.device)
                layer_loss = self.layer_penalty(tmp) * layer_loss
                cumulative_loss += layer_loss
                loss_by_depths.insert(0, layer_loss.detach().cpu().numpy()
                )
            else:
                raise NotImplementedError('Unknown loss')
            _, unique_indices = unique(layer_labels, dim=0)
            # max_loss_lower_layer = torch.max(
            #     max_loss_lower_layer.to(layer_loss.device), layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
        # return cumulative_loss / labels.shape[1], loss_by_depths, mask_by_depths
        # print(loss_by_depths[0].shape, mask_by_depths[0].shape)
        return cumulative_loss / labels.shape[1], np.array(loss_by_depths) / labels.shape[1]

    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    """
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device='cuda'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device(self.device)
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # TODO shouldn't be one
            labels = labels.contiguous().view(-1, 1) # [batch_size, 1]
            if labels.shape[0] != batch_size:
                raise ValueError(f'Num of labels does not match num of features, {labels.shape[0]} != {batch_size}')
            # compute the 2d mask
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # features: [bsz, n_views, f_dim]
        # mask.shape = [batch_size, batch_size]

        # PROBLEMS!! Not fixed label size
        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # [batch_size * n_views, feature_size]
        
        self.contrast_mode = 'all'

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print("Fuck")
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # print("hello")
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # [batch_size * n_views, batch_size * n_views]
        # print("yoooooo")
        # mask-out self-contrast cases (the diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        eplison = 1e-8
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # [batch_size * n_views, batch_size * n_views]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eplison) # [batch_size * n_views, batch_size * n_views]
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eplison) # [batch_size * n_views]
        # print(mean_log_prob_pos)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(loss)
        return loss
    
def pair_cosine_similarity(self, x, x_adv, eps=1e-8):
        n = x.norm(p=2, dim=1, keepdim=True)
        n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
        # print(2, x @ x.t())
        # print(x)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (
                x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)
    # clamp方法：对输入参数按照自定义的范围进行裁剪，最后将参数裁剪的结果作为输出。所以输入参数一共有三个，分别是需要进行裁剪的 Tensor 数据类型的变量、裁剪的上边界和裁剪的下边界。
    # 具体的裁剪过程是：使用变量中的每个元素分别和裁剪的上边界及裁剪的下边界的值进行比较，如果元素的值小于裁剪的下边界的值，该元素就被重写成裁剪的下边界的值；同理,如果元素的值大于裁剪的上边界的值，该元素就被重写成裁剪的上边界的值。

def nt_xent(self, x, x_adv, mask, mask_neg, mask_neg_2, mask_positive, cuda=True, t=0.1):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    # print(1, x.shape, x_adv.shape, x_c.shape)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).float()
    # print(3, mask, mask_count, mask_reverse)
    # print(4, mask - torch.eye(x.size(0)).long())
    # x * (mask - torch.eye(x.size(0)).long())和非自己本身的向量的相似计算，但是同属一个类的
    # x_c * mask 在正例增强里面，尽管也是同一个类，但是并不是自己本身，所以是乘mask
    # x.sum(1)代表x每个anchor所有的相似度相加
    # 对于某个example来说，是（所有的正例*增强数据相加/所有例子之和（在这里设置负权重）+增强数据也算一遍）/正例数量
    # print(8, x_c * mask)
    # print(5, x * (mask - torch.eye(x.size(0)).long()) + x_c * mask)
    # print(6, x.sum(1) + x_c.sum(1) - torchlr.exp(torch.tensor(1 / t)))
    # print(7, (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (
    #             x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))))
    # print(torch.norm(x,p=2),torch.norm(x_adv,p=2),torch.norm(x_c,p=2),torch.norm(mask,p=2),torch.norm(mask_neg,p=2),torch.norm(mask_reverse,p=2),
    #     '第一次' )
    # print(x * (mask - torch.eye(x.size(0)).float().cuda()) + x_c * mask,'第2次')
    # print( (x* mask_neg).sum(1)  + (x_c* mask_neg_2).sum(1)  - torch.exp(torch.tensor(1 / t)).cuda(),'第3次')
    # print( (x* mask_neg).sum(1)  + (x_c* mask_neg_2).sum(1) ,'第7次')
    # print(x,x_c,x*mask_neg,x_c*mask_neg_2,'第8次')
    # diswei=(x * (mask - torch.eye(x.size(0)).float().cuda()) + x_c * mask) / (
    #                 (x* mask_neg).sum(1)+ (x_c* mask_neg).sum(1) ) + mask_reverse
    # print(torch.log(diswei).sum(1),'第9次')
    # print(diswei,torch.log(diswei),'第10次')
    if cuda:
        # #print(x.device,mask.device,x_c.device,mask_neg.device,mask_reverse.device,torch.exp(torch.tensor(1 / t)).device)
        # # dis = (x * (mask - torch.eye(x.size(0)).float().cuda()) + x_c * mask) / (
        # #             (x* mask_neg).sum(1)+ (x_c* mask_neg).sum(1) - torch.exp(torch.tensor(1 / t)).cuda()) + mask_reverse # 1/t是不是也应该加个neg，
        # #             这里加mask_reverse的意思也没搞懂(是为了log的时候弄成0），删掉了源码里面有的1/t，不然会造成数据为inf
        # dis = (x * (mask_positive - 1.4*torch.eye(x.size(0)).float().cuda()) + x_c * mask_positive) / (
        #             (x* mask_neg).sum(1)+ (x_c* mask_neg_2).sum(1) ) + mask_reverse # 1/t是不是也应该加个neg，这里加mask_reverse的意思也没搞懂(是为了log的时候弄成0）
        # #这里还有一个问题，对于adv来说正例是1，负例是1.2,对于负例来说，也不能包含自己本身。
        # dis_adv = (x_adv * (mask_positive - 1.4*torch.eye(x.size(0)).float().cuda()) + x_c.T * mask_positive) / (
        #         (x_adv* mask_neg).sum(1)  + (x_c* mask_neg_2).sum(0)  ) + mask_reverse#adv为什么要转置一下呢# 这里使用positive层级变化 这里不确定是否增强数据也用于对比学习，比如translation翻译的数据 先舍弃掉
        #从这里开始正常分割
        dis = torch.div((x * (mask_positive - self.args.con1* torch.eye(x.size(0)).float().cuda()) + x_c * mask_positive), (
                (x * mask_neg).sum(1, keepdim=True).repeat([1, x.size(0)]) + (x_c * mask_neg_2).sum(1,
                                                                                                    keepdim=True).repeat(
            [1, x.size(0)]))) + mask_reverse # 1/t是不是也应该加个neg，这里加mask_reverse的意思也没搞懂(是为了log的时候弄成0）
        # 这里还有一个问题，对于adv来说正例是1，负例是1.2,对于负例来说，也不能包含自己本身。
        # dis_adv = torch.div(
        #     (x_adv * (mask_positive - self.args.con1 * torch.eye(x.size(0)).float().cuda()) + x_c.T * mask_positive), (
        #             (x_adv * mask_neg).sum(1, keepdim=True).repeat([1, x.size(0)]) + (x_c * mask_neg_2).sum(
        #         0).unsqueeze(1).repeat([1, x.size(
        #         0)]))) + mask_reverse # adv为什么要转置一下呢# 这里使用positive层级变化 这里不确定是否增强数据也用于对比学习，比如translation翻译的数据 先舍弃掉
        #没有数据增强
        # dis = torch.div(x * (mask_positive - self.args.con1* torch.eye(x.size(0)).float().cuda()) ,
        #        (x * mask_neg).sum(1, keepdim=True).repeat([1, x.size(0)]).clamp(min=1e-8) ) + mask_reverse + torch.eye(x.size(0)).float().cuda()# 1/t是不是也应该加个neg，这里加mask_reverse的意思也没搞懂(是为了log的时候弄成0）
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (
                (x * mask_neg).sum(1) + (x_c * mask_neg_2).sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (
                (x_adv * mask_neg).sum(1) + (x_c * mask_neg_2).sum(0) - torch.exp(
            torch.tensor(1 / t))) + mask_reverse
    # print(dis,torch.log(dis),mask_count,mask_reverse,'第4次')
    # print(torch.log(dis).sum(1),torch.log(dis_adv).sum(1),'第5次')
    #loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    #没有数据增强
    loss = (torch.log(dis).sum(1) ) / mask_count

    return loss

class WeightedCosineSimilarityLoss(nn.Module):
    """Docstring for WeightedCosineSimilarityLoss."""

    def __init__(self, class_num: int):
        super(WeightedCosineSimilarityLoss, self).__init__()
        self.class_num = class_num

    def forward(self, features, labels=None):
        """
        :features (batch_size, num_labels, feature_dim)
        :labels (batch_size, num_labels)
        """
        device = features.device
        temp_labels = labels
        loss = torch.tensor(0, device=device, dtype=torch.float64)
        for i in range(self.class_num):
            pos_idx = torch.where(temp_labels[:, i] == 1)[0] # (pos_num, 1)
            if len(pos_idx) == 0:
                continue
            neg_idx = torch.where(temp_labels[:, i] != 1)[0] # (neg_num, 1)
            pos_samples = features[pos_idx, i, :].squeeze(1) # (pos_num, feature_dim)
            neg_samples = features[neg_idx, i, :].squeeze(1) # (neg_num, feature_dim)
            size = neg_samples.shape[0] + 1

            temp_labels = temp_labels.float()
            dist = self.hamming_distance_by_matrix(temp_labels) # (num_labels, num_labels)
            pos_weight = 1 - dist[pos_idx, :][:, pos_idx] / self.class_num # (pos_num, pos_num)
            neg_weight = dist[pos_idx, :][:, neg_idx] # (pos_num, neg_num)
            pos_dis = self.exp_cosine_sim(pos_samples, pos_samples) * pos_weight
            neg_dis = self.exp_cosine_sim(pos_samples, neg_samples) * neg_weight
            denominator = neg_dis.sum(1) + pos_dis
            loss += torch.mean(torch.log(denominator / (pos_dis * size)))
        return loss

    def hamming_distance_by_matrix(self, labels):
        return torch.matmul(labels, (1 - labels).T) + torch.matmul(1 - labels, labels.T)

    def exp_cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.exp(
            torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)
        )
