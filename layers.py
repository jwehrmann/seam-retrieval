import torch
import torch.nn as nn


def l2norm(X):
    """L2-normalize columns of X
    """    
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X)    
    X = torch.div(X, a)    
    return X


def global_initializer(module):

    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight.data)
            # m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.uniform_(-0.1, 0.1)
