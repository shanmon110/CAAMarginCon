"""
Author: LI Zhe (lizhe.li@connect.polyu.hk)
Date: Oct 07, 2022
"""
from __future__ import print_function

import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.m = 0.2
        self.s = 10
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.weight = torch.nn.Parameter(torch.FloatTensor(2793, 2793), requires_grad=False)
        self.v = torch.nn.Parameter(torch.FloatTensor(2793, 2793), requires_grad=True)
        self.fc = nn.Linear(192, 2793)
        self.ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, labels=None, mask=None):
        """

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

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
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        x_norm = torch.norm(contrast_feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x = torch.div(contrast_feature, x_norm)
        w_norm = torch.norm(anchor_feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        w = torch.div(anchor_feature, w_norm)
        cosine = torch.matmul(x, w.T)
        cosine = F.normalize(cosine, p=2, dim=1)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        output = self.s * phi
        output = output.double()
#        output = F.normalize(output, p=2, dim=1)
        
        similarity = x_norm * w_norm.T * (phi + cosine)
        similarity2 = torch.matmul(anchor_feature, contrast_feature.T)
               
        attlabels = torch.cat((labels, labels), dim=0)

        # embd is the weights of the FC layer for classification R^(dxC)
        embd = F.normalize(self.weight, p=2, dim=0)  # normalize the embedding
        h1 = self.fc(contrast_feature)
        h2 = torch.tanh(h1)
        one_hot = torch.zeros_like(embd)
        one_hot.scatter_(1, attlabels.view(-1, 1), 1)
#        temp = torch.matmul(h2, one_hot)        
        denominator = torch.exp(torch.mm(h2, one_hot)) 

        A = []  # attention corresponding to each feature fector
        n = contrast_feature.shape[0]
        for i in range(n):
            a_i = denominator[i][attlabels[i]] / torch.sum(denominator[i])
            A.append(a_i)
            # a_i's
        atten_class = torch.stack(A)
        atten_class = F.normalize(atten_class, p=2, dim=1)
       # atten_class = data_normal(atten_class)

        #        #a_ij's
        A = torch.min(atten_class.expand(n, n),
                      atten_class.view(-1, 1).expand(n, n))  # pairwise minimum of attention weights

        # compute logits
        
        anchor_dot_contrast = torch.div(
            output,
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        logits = logits * A

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        mask = mask * logits_mask

        # compute log_prob

        exp_logits = (torch.exp(logits)) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        return loss
