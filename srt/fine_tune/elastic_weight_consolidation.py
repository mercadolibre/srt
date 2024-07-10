from copy import deepcopy
import torch
from torch import nn

from srt.progress import progress


class EWCPenalty:
    def __init__(self, importance_weight):
        self.importance_weight = importance_weight
        self.precision_matrices_ = self.means_ = None

    def fit(self, pretrained_model, train_data, n_batches):
        params = {n: p for n, p in pretrained_model.named_parameters() if p.requires_grad}
        means = {}

        for n, p in deepcopy(params).items():
            means[n] = p.data

        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        for i, batch in progress(enumerate(train_data), desc='fitting ewc', tot=n_batches):
            if i == n_batches: break
            pretrained_model.zero_grad()
            loss = pretrained_model.calculate_loss(batch.to('cuda'))
            loss.backward()

            for n, p in pretrained_model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / n_batches

        self.means_ = means
        self.precision_matrices_ = precision_matrices
        return self

    def save(self, fname):
        with open(fname, 'wb') as f:
            torch.save({
                'means': self.means_,
                'precision_matrices': self.precision_matrices_,
                'importance_weight': self.importance_weight}, f)

    @classmethod
    def load(cls, fname):
        d = torch.load(fname)
        res = cls(d['importance_weight'])
        res.precision_matrices_ = d['precision_matrices']
        res.means_ = d['means']
        return res

    def get_penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.precision_matrices_[n] * (p - self.means_[n]) ** 2
            loss += _loss.sum()
        return loss * self.importance_weight
